package com.enigmapattern.macaque.dataproviders.output

import com.enigmapattern.macaque.dataproviders.UnexpectedDataShapeException
import com.enigmapattern.macaque.results.BoundBox
import com.enigmapattern.macaque.results.DetectionResult
import java.lang.Exception
import java.nio.ByteBuffer

/**
 * Output Data Provider for Object Detectors which have 4 default output tensors (locations, classes,
 * scores and number of detections) and are returning bounding boxes
 */
class Detector4outputsBoundingBoxesDataProvider: OutputDataProvider() {

    /**
     * Method for parsing and returning results of detection.
     *
     * It operates on outputsHashMap and with use of outputTensorsShapes parses it and provides list
     * of detection results.
     * @return List of DetectionResult
     */
    override fun getResults(): List<DetectionResult> {
        //locations
        val results = mutableListOf<DetectionResult>()

        val numDetections: Int
        val locationsBuffer: ByteBuffer
        val classesBuffer: ByteBuffer
        val scoresBuffer: ByteBuffer

        try {
            numDetections = outputTensorsShapes!![0][1]

            locationsBuffer = (outputsHashMap as HashMap<Int, ByteBuffer>)[0]!!
            classesBuffer = (outputsHashMap as HashMap<Int, ByteBuffer>)[1]!!
            scoresBuffer = (outputsHashMap as HashMap<Int, ByteBuffer>)[2]!!
        } catch (e: Exception){
            throw UnexpectedDataShapeException("The output tensors shapes are different than expected")
        }
        locationsBuffer.rewind()
        classesBuffer.rewind()
        scoresBuffer.rewind()

        for (detection in 0 until numDetections){
            val ymin = locationsBuffer.float
            val xmin = locationsBuffer.float
            val ymax = locationsBuffer.float
            val xmax = locationsBuffer.float

            val cls = classesBuffer.float
            val score = scoresBuffer.float

            results.add(
                DetectionResult(
                    cls.toInt(),
                    score,
                    BoundBox(
                        xmin,
                        xmax,
                        ymin,
                        ymax
                    )
                )
            )

        }
        return results
    }

}