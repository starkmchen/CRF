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

package org.apache.spark.examples.ml

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.nlp.ConditionalRandomField
import org.apache.spark.mllib.nlp.CRF

/**
 * An example demonstrating a CRF.
 * Run with
 * {{{
 * bin/run-example ml.ConditionalRandomFieldExample <modelFile> <featureFile>
 * }}}
 */

object ConditionalRandomFieldExample {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      // scalastyle:off println
      System.err.println("Usage: ml.CRFExample <modelFile> <featureFile> <testFile>")
      // scalastyle:on println
      System.exit(1)
    }
    val template = args(0)
    val feature = args(1)
    val test = args(2)

    // Creates a Spark context and a SQL context
    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}")
      .set(s"spark.yarn.jar", s"/home/hujiayin/git/spark/yarn/target/spark.yarn.jar")

    val sc = new SparkContext(conf)

    val rowRDD = sc.textFile(template).filter(_.nonEmpty)
    val rowRddF = sc.textFile(feature).filter(_.nonEmpty)

    val crf = new ConditionalRandomField()
    val model = crf.trainRdd(rowRDD, rowRddF, sc)

    val modelPath = "/home/hujiayin/git/CRFConfig/CRFOutput"
    model.save(sc, modelPath)

    val rowRddT = sc.textFile(test).filter(_.nonEmpty)
    val modelRDD = sc.parallelize(model.load(sc, modelPath).CRFSeries(0))
    val newResult = CRF.verifyCRF(rowRddT, modelRDD, sc)
    var idx: Int = 0
    var i: Int = 0
    var temp: String = ""
    while (i < newResult.CRFSeries.length) {
      while (idx < newResult.CRFSeries(i).length) {
        temp += newResult.CRFSeries(i)(idx)
        if ((idx + 1) % 3 == 0) {
          // scalastyle:off println
          println(temp)
          // scalastyle:on println
          temp = ""
        }
        idx += 1
      }
      idx = 0
      i += 1
    }
    sc.stop()
  }

}
