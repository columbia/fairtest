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

package org.apache.spark.mllib.tree.impurity
import org.apache.spark.mllib.tree.impurity.Measure._
import org.apache.spark.mllib.tree.impurity.Aggregator._
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import scala.math.{log,pow,sqrt,max}

/**
 * :: Experimental ::
 * Class for calculating [[http://en.wikipedia.org/wiki/Binary_entropy_function entropy]] during
 * binary classification.
 */
@Experimental
case class Bias(val dim: (Int, Int), val measure: Measure, val aggregator: Aggregator, val conf: Double) extends Impurity {

  /**
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @DeveloperApi
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val numClasses = counts.length
    var mi = 0.0
    var mi_sq = 0.0

    var freqs1 = new Array[Double](dim._1)
    var freqs2 = new Array[Double](dim._2)

    var i = 0
    var j = 0
    for (i <- 0 until dim._1; j <- 0 until dim._2) {
        val count = counts(i*dim._2 + j)
        freqs1(i) += count / totalCount
        freqs2(j) += count / totalCount
    }

    i = 0
    j = 0
    for (i <- 0 until dim._1; j <- 0 until dim._2) {
        val count = counts(i*dim._2 + j)

        if ((count != 0) && (freqs1(i) != 0) && (freqs2(j) != 0)) {
            val freq = count / totalCount
            mi += freq * (log(freq) - log(freqs1(i)) - log(freqs2(j)))
            mi_sq += freq * pow((log(freq) - log(freqs1(i)) - log(freqs2(j))), 2)
        }
    }
    
    val std_dev = sqrt((mi_sq - pow(mi, 2))/totalCount)
    val mi_inf = max(mi - 1.96*std_dev/sqrt(totalCount), 0.0)
    
    mi
  }
  
  def calcuate_gain(current: Double, left_measure: Double, right_measure: Double, left_weight:Double, right_weight: Double) : Double = aggregator match {
    case WeightedAvg => left_weight * left_measure + right_weight * right_measure
    case Avg => (left_measure + right_measure) / 2.0
    case Max => max(left_measure, right_measure)
    case _ => throw new IllegalArgumentException(s"Did not recognize Bias Aggregator: $aggregator")
  } 

  /**
   * :: DeveloperApi ::
   * variance calculation
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @DeveloperApi
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("Bias.calculate")

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  def instance: this.type = this

}

/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param numClasses  Number of classes for label.
 */
private[tree] class BiasAggregator(numClasses: Int, dim: (Int, Int), measure: Measure, aggregator: Aggregator, conf: Double)
  extends ImpurityAggregator(numClasses) with Serializable {

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, instanceWeight: Double): Unit = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"BiasAggregator given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"BiasAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    allStats(offset + label.toInt) += instanceWeight
  }

  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): BiasCalculator = {
    new BiasCalculator(allStats.view(offset, offset + statsSize).toArray, dim, measure, aggregator, conf)
  }

}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[BiasAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[tree] class BiasCalculator(stats: Array[Double], dim: (Int, Int), measure: Measure, aggregator: Aggregator, conf: Double)
  extends ImpurityCalculator(stats) {

  /**
   * Make a deep copy of this [[ImpurityCalculator]].
   */
  def copy: BiasCalculator = new BiasCalculator(stats.clone(), dim, measure, aggregator, conf)

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double = new Bias(dim, measure, aggregator, conf).calculate(stats, stats.sum)

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long = stats.sum.toLong

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  /**
   * Probability of the label given by [[predict]].
   */
  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"BiasCalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "Bias does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"BiasCalculator(stats = [${stats.mkString(", ")}])"

}
