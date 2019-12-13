package com.enigmapattern.macaque.dataproviders.output

import com.enigmapattern.macaque.results.ClassificationResult
import java.nio.ByteBuffer

/**
 * Output Data Provider for Classifier models which have single output tensor with classification
 * scores for each class
 */
class ClassifierDataProvider: OutputDataProvider() {

    /**
     * Method for parsing and returning results of classification.
     *
     * It operates on outputsHashMap and returns list with single result.
     * @return Single element list of ClassificationResult
     */
    override fun getResults(): List<ClassificationResult> {
        with(outputsHashMap!![0] as ByteBuffer){

            this.rewind()

            var maxIdx = -1
            var maxVal = java.lang.Float.NEGATIVE_INFINITY
            var idx = 0

            while(this.hasRemaining()){
                val next = this.float
                if (next > maxVal) {
                    maxVal = next
                    maxIdx = idx
                }
                idx += 1
            }

            return listOf(
                ClassificationResult(
                    maxIdx,
                    maxVal
                )
            )
        }
    }
}