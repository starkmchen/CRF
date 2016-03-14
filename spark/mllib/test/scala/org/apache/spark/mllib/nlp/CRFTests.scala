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

import java.io.Serializable

import org.apache.spark.{SparkConf, SparkContext, Logging, SparkFunSuite}

/**
 * The language source files could be found at
 * http://www.cnts.ua.ac.be/conll2000/chunking/
 */

class CRFTests extends SparkFunSuite with Logging with Serializable {
  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("MLlibUnitTest")
  @transient var sc: SparkContext = new SparkContext(conf)
  val dic = Array(
    "gram",
    "U00:%x[-2,0]",
    "U01:%x[-1,0]",
    "U02:%x[0,0]",
    "U03:%x[1,0]",
    "U04:%x[2,0]",
    "U05:%x[-1,0]/%x[0,0]",
    "U06:%x[0,0]/%x[1,0]",
    "U10:%x[-2,1]",
    "U11:%x[-1,1]",
    "U12:%x[0,1]",
    "U13:%x[1,1]",
    "U14:%x[2,1]",
    "U15:%x[-2,1]/%x[-1,1]",
    "U16:%x[-1,1]/%x[0,1]",
    "U17:%x[0,1]/%x[1,1]",
    "U18:%x[1,1]/%x[2,1]",
    "U20:%x[-2,1]/%x[-1,1]/%x[0,1]",
    "U21:%x[-1,1]/%x[0,1]/%x[1,1]",
    "U22:%x[0,1]/%x[1,1]/%x[2,1]",
    "# Bigram",
    "B"
  )

  val template = sc.parallelize(dic)

  val file = Array(
    "He|PRP|B-NP\t"+
    "reckons|VBZ|B-VP\t"+
    "the|DT|B-NP\t"+
    "current|JJ|I-NP\t"+
    "account|NN|I-NP\t"+
    "deficit|NN|I-NP\t"+
    "will|MD|B-VP\t"+
    "narrow|VB|I-VP\t"+
    "to|TO|B-PP\t"+
    "only|RB|B-NP\t"+
    "#|#|I-NP\t"+
    "1.8|CD|I-NP\t"+
    "billion|CD|I-NP\t"+
    "in|IN|B-PP\t"+
    "September|NNP|B-NP\t"+
    ".|.|O",
    "He|PRP|B-NP\t"+
    "reckons|VNZ|B-VP"
  )

  val modelPath = "/home/hujiayin/git/CRFConfig/CRFModel"
  val resultPath = "/home/hujiayin/git/CRFConfig/CRFResult"
  val src = sc.parallelize(file).cache()
  val CRFModel = CRF.runCRF(template, src, sc)
  CRFModel.save(sc,modelPath)
  val modelRDD = sc.parallelize(CRFModel.load(sc,modelPath).CRFSeries(0))
  val result = CRF.verifyCRF(src, modelRDD, sc)
  result.save(sc,resultPath)
  var idx: Int = 0
  var i: Int = 0
  var temp: String = ""
  println("Word|WordCategory|Label|PredictiveLabel")
  println("---------------------------------------")
  while(idx < result.CRFSeries.length) {
    while(i < result.CRFSeries(idx).length) {
      temp += result.CRFSeries(idx)(i)
      if ((i + 1) % 4 == 0) {
        // scalastyle:off println
        println(temp)
        // scalastyle:on println
        temp = ""
      }
      i += 1
    }
    i = 0
    idx += 1
  }

  val newFile = Array(
    "Confidence|NN\t"+
    "in|IN\t"+
    "the|DT\t"+
    "pound|NN\t"+
    "is|VBZ\t"+
    "widely|RB\t"+
    "expected|VBN\t"+
    "to|TO\t"+
    "take|VB\t"+
    "another|DT\t"+
    "sharp|JJ\t"+
    "dive|NN\t"+
    "if|IN\t"+
    "trade|NN\t"+
    "figures|NNS\t"+
    "for|IN\t"+
    "September|NNP"
  )

  val newSrc = sc.parallelize(newFile).cache()
  val newResult = CRF.verifyCRF(newSrc, modelRDD, sc)
  idx = 0
  i = 0
  temp = ""
  while(idx < newResult.CRFSeries.length) {
    while(i < newResult.CRFSeries(idx).length) {
      temp += newResult.CRFSeries(idx)(i)
      if ((i + 1) % 3 == 0) {
        // scalastyle:off println
        println(temp)
        // scalastyle:on println
        temp = ""
      }
      i += 1
    }
    i = 0
    idx += 1
  }

}
