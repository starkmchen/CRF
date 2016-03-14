/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.nlp

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

private[mllib] class CRF extends Serializable {
  private val freq: Int = 1
  private val maxiter: Int = 100000
  private val cost: Double = 1.0
  private val eta: Double = 0.0001
  private val C: Float = 1
  private val threadNum: Int = Runtime.getRuntime.availableProcessors()
  private val threadPool: Array[CRFThread] = new Array[CRFThread](threadNum)
  private var featureIdx: FeatureIndex = new FeatureIndex()
  @transient var sc: SparkContext = _


  /**
   * Internal method to verify the CRF model
   * @param test the same source in the CRFLearn
   * @param models the output from CRFLearn
   * @return the source with predictive labels
   */
  def verify(test: RDD[String],
             models: RDD[String],
             sc: SparkContext): RDD[Array[String]] = {
    val tagger: Tagger = new Tagger()
    featureIdx = featureIdx.openFromArray(models)
    val taggerList: Array[Tagger] = tagger.read(test, featureIdx)
    var i: Int = 0
    val taggerX: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]
    while (i < taggerList.length) {
      taggerList(i).mode = 1
      taggerList(i).parse()
      taggerX.append(taggerList(i).createOutput())
      i += 1
    }
    val modelRdd = sc.parallelize(taggerX)
    modelRdd
  }

  /**
   * Internal method to train the CRF model
   * @param templates the template to train the model
   * @param trains the source for the training
   * @return the model of the source
   */
  def learn(templates: RDD[String],
            trains: RDD[String],
            sc: SparkContext): RDD[String] = {
    val tagger: Tagger = new Tagger()
    val template: Array[String] = templates.toLocalIterator.toArray
    featureIdx.openTemplate(template)
    featureIdx = featureIdx.openTagSet(trains)
    val taggerList: Array[Tagger] = tagger.read(trains, featureIdx)
    var i: Int = 0

    while (i < taggerList.length) {
      taggerList(i).mode = 2
      featureIdx.buildFeatures(taggerList(i))
      featureIdx.shrink(freq)
      featureIdx.initAlpha(featureIdx.maxid)
      i += 1
    }
    featureIdx.getFeatureIndexHeader
    featureIdx.setFeatureIdx(taggerList)
    featureIdx.setBaseIdx(taggerList)
    val taggerListRdd = sc.parallelize(taggerList).cache()
    var alphaRdd = sc.parallelize(featureIdx.alpha)
    alphaRdd = runsCRF(taggerListRdd, featureIdx, alphaRdd, sc)
    val modelRdd = sc.parallelize(featureIdx.saveModel(false, alphaRdd))
    modelRdd
  }

  def runsCRF(taggerRdd: RDD[Tagger], featureIndex: FeatureIndex,
              alphaRdd: RDD[Double], sc: SparkContext): RDD[Double] = {
    var diff: Double = 0.0
    var old_obj: Double = 1e37
    var converge: Int = 0
    var itr: Int = 0
    var all: Int = 0
    val opt = new Optimizer()
    var i: Int = 0
    var k: Int = 0
    val expected: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    var err: Int = 0
    var zeroOne: Int = 0
    var obj: Double = 0.0
    var idx: Int = 0

    val tag: Array[Tagger] = taggerRdd.toLocalIterator.toArray
    val tagger: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    val alp: Array[Double] = alphaRdd.toLocalIterator.toArray
    val alpha: ArrayBuffer[Double] = new ArrayBuffer[Double]()

    alpha.appendAll(alp)
    tagger.appendAll(tag)

    while (i < tagger.length) {
      all += tagger(i).x.size
      i += 1
    }
    i = 0
    while (i < featureIdx.maxid) {
      expected.append(0.0)
      i += 1
    }
    i = 0

    while (itr <= maxiter) {
      while (i < featureIdx.maxid) {
        expected.update(i, 0.0)
        i += 1
      }
      i = 0
      idx = 0
      obj = 0
      zeroOne = 0
      err = 0
      while (idx >= 0 && idx < tagger.size) {
        // printf("Round=%d\n", idx)
        // printf("Before=====expected[0]=%2.5f, expected[1]=%2.5f,
        // expected[125]=%2.5f\n", expected.head, expected(1), expected(125))
        obj += tagger(idx).gradient(expected, alpha)
        // printf("After=====expected[0]=%2.5f, expected[1]=%2.5f,
        // expected[125]=%2.5f\n", expected.head, expected(1), expected(125))
        err += tagger(idx).eval()
        if (err != 0) {
          zeroOne += 1
        }
        idx = idx + 1
      }
      while (k < featureIndex.maxid) {
        obj += (alpha(k) * alpha(k) / (2.0 * C))
        expected(k) += alpha(k) / C
        k += 1
      }
      k = 0
      i = 0
      if (itr == 0) {
        diff = 1.0
      } else {
        diff = math.abs((old_obj - obj) / old_obj)
      }

      // printf("BeforeLBFGS===alpha(0)=%2.5f, expected(0)=%2.5f,expected(1)=%2.5f," +
      // "expected(125)=%2.5f, obj=%2.5f\n", alpha.head,
      // expected.head, expected(1), expected(125), obj)

      opt.optimizer(featureIndex.maxid, alpha, obj, expected, C)
      // printf("iter=%d, terr=%2.5f, serr=%2.5f, act=%d, obj=%2.5f,diff=%2.5f\n",
      // itr, 1.0 * err / all, 1.0 * zeroOne / tagger.size, featureIndex.maxid, obj, diff)

      // printf("AfterLBFGS===alpha(0)=%2.5f, expected(0)=%2.5f,expected(1)=%2.5f," +
      // "expected(125)=%2.5f, obj=%2.5f\n", alpha.head,
      // expected.head, expected(1), expected(125), obj)

      old_obj = obj

      if (diff < eta) {
        converge += 1
      } else {
        converge = 0
      }
      if (converge == 3) {
        itr = maxiter + 1 // break
      }
      if (diff == 0) {
        itr = maxiter + 1 // break
      }

      itr += 1
    }
    sc.parallelize(alpha)
  }

  /**
   * Parse segments in the unit sentences or paragraphs
   * @param taggerRdd the tagger in the template
   * @param featureIndex the index of the feature
   * @param alphaRdd the model
   */

  def runCRF(taggerRdd: RDD[Tagger], featureIndex: FeatureIndex,
             alphaRdd: RDD[Double], sc: SparkContext): RDD[Double] = {
    var diff: Double = 0.0
    var old_obj: Double = 1e37
    var converge: Int = 0
    var itr: Int = 0
    var all: Int = 0
    val opt = new Optimizer()
    var i: Int = 0
    var k: Int = 0

    val tag: Array[Tagger] = taggerRdd.toLocalIterator.toArray
    val tagger: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    val alp: Array[Double] = alphaRdd.toLocalIterator.toArray
    val alpha: ArrayBuffer[Double] = new ArrayBuffer[Double]()

    alpha.appendAll(alp)
    tagger.appendAll(tag)

    while (i < tagger.length) {
      all += tagger(i).x.size
      i += 1
    }
    i = 0

    while (itr <= maxiter) {
      while (i < threadNum) {
        threadPool(i) = new CRFThread()
        threadPool(i).start_i = i
        threadPool(i).size = tagger.size
        threadPool(i).x = tagger
        threadPool(i).start()
        i += 1
      }
      threadPool(0).alp = alpha
      i = 0
      while (i < threadNum) {
        threadPool(i).join()
        i += 1
      }
      i = 0
      while (i < threadNum) {
        if (i > 0) {
          threadPool(0).obj += threadPool(i).obj
          threadPool(0).err += threadPool(i).err
          threadPool(0).zeroOne += threadPool(i).zeroOne
        }
        while (k < featureIndex.maxid) {
          if (i > 0) {
            threadPool(0).expected(k) += threadPool(i).expected(k)
          }
          threadPool(0).obj += (alpha(k) * alpha(k) / (2.0 * C))
          threadPool(0).expected(k) += alpha(k) / C
          // printf("ALPHA(k)=%2.5f ",alpha(k))
          k += 1
        }
        // printf("\n")
        k = 0
        i += 1
      }
      i = 0
      if (itr == 0) {
        diff = 1.0
      } else {
        diff = math.abs((old_obj - threadPool(0).obj) / old_obj)
      }
      old_obj = threadPool(0).obj
      printf("iter=%d, terr=%2.5f, serr=%2.5f, act=%d, obj=%2.5f,diff=%2.5f\n",
        itr, 1.0 * threadPool(0).err / all,
        1.0 * threadPool(0).zeroOne / tagger.size, featureIndex.maxid,
        threadPool(0).obj, diff)
      if (diff < eta) {
        converge += 1
      } else {
        converge = 0
      }
      if (converge == 3) {
        itr = maxiter + 1 // break
      }
      if (diff == 0) {
        itr = maxiter + 1 // break
      }
      // printf("BeforeLBFGS===alpha(0)=%2.5f,threadPool(0)." +
      // "expected(0)=%2.5f,threadPool(0).expected(1)=%2.5f," +
      // "threadPool(0).expected(125)=%2.5f, threadPool(0).obj=%2.5f\n",
      // threadPool(0).alp(0),
      // threadPool(0).expected(0), threadPool(0).expected(1),
      // threadPool(0).expected(125), threadPool(0).obj)*/
      opt.optimizer(featureIndex.maxid, threadPool(0).alp,
        threadPool(0).obj, threadPool(0).expected, C)
      // printf("AfterLBFGS===alpha(0)=%2.5f,threadPool(0)." +
      // "expected(0)=%2.5f,threadPool(0).expected(1)=%2.5f,threadPool(0)." +
      // "expected(125)=%2.5f, threadPool(0).obj=%2.5f\n", threadPool(0).alp(0),
      // threadPool(0).expected(0), threadPool(0).expected(1),
      // threadPool(0).expected(125), threadPool(0).obj)*/
      // threadPool(0).alp.appendAll(alpha)
      // featureIdx.setAlpha(alpha)
      // alpha = rtnVal.x
      // threadPool(0).obj = rtnVal.f
      // threadPool(0).expected = rtnVal.g
      itr += 1
    }
    sc.parallelize(alpha)
  }

  /**
   * Use multiple threads to parse the segments
   * in a unit sentence or paragraph.
   */
  class CRFThread extends Thread {
    var x: ArrayBuffer[Tagger] = null
    var start_i: Int = 0
    var err: Int = 0
    var zeroOne: Int = 0
    var size: Int = 0
    var obj: Double = 0.0
    var expected: ArrayBuffer[Double] = new ArrayBuffer[Double]()
    var alp: ArrayBuffer[Double] = new ArrayBuffer[Double]()

    def initExpected(): Unit = {
      var i: Int = 0
      while (i < featureIdx.maxid) {
        expected.append(0.0)
        i += 1
      }
    }

    /**
     * Train CRF model and calculate the expectations
     */
    override def run: Unit = {
      var idx: Int = 0
      initExpected()
      while (idx >= start_i && idx < size) {
        // printf("Round=%d\n", idx)
        obj += x(idx).gradient(expected, threadPool(0).alp)
        // printf("expected[0]=%2.5f, expected[1]=%2.5f,
        // expected[125]=%2.5f\n", expected(0), expected(1), expected(125))
        err += x(idx).eval()
        if (err != 0) {
          zeroOne += 1
        }
        idx = idx + 1
      }
    }
  }

}

@DeveloperApi
object CRF {
  /**
   * Train CRF Model
   * Feature file format
   * word|word characteristic|designated label
   *
   * @param templates Source templates for training the model
   * @param features Source files for training the model
   * @return Model of a unit
   */
  def runCRF(templates: RDD[String],
             features: RDD[String],
             sc: SparkContext): CRFModel = {

    val crf = new CRF()
    val result: Array[String] = crf.learn(templates, features, sc)
      .toLocalIterator.toArray
    val finalArray: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]()
    finalArray.append(result)
    new CRFModel(finalArray.toArray)
  }

  /**
   * Verify CRF model
   * Test result format:
   * word|word characteristic|designated label|predicted label
   *
   * @param tests  Source files to be verified
   * @param models Model files after call the CRF learn
   * @return Source files with the predictive labels
   */
  def verifyCRF(tests: RDD[String],
                models: RDD[String],
                sc: SparkContext): CRFModel = {
    val crf = new CRF()
    val result: Array[Array[String]] = crf.verify(tests, models, sc).toLocalIterator.toArray
    new CRFModel(result)
  }
}
