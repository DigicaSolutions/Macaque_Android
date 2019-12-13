package com.enigmapattern.macaque.results

import android.graphics.RectF

/**
 * Class representing single result of detection.
 *
 * Representation of single object detected by object detection model.
 *
 * @property cls class of detected object
 * @property score confidence with which object was detected
 * @property bbox BoundBox object representing location of detected object on analyzed image
 */
class DetectionResult (val cls: Int, val score: Float, val bbox: BoundBox){
}

/**
 * Bounding Box class
 *
 * Object representing bounding box - location of detected object on analyzed image.
 * Coordinates are in normalized scale between 0 and 1.
 *
 * @property xmin left edge of the bounding box
 * @property xmax right edge of the bounding box
 * @property ymin bottom edge of the bounding box
 * @property ymax top edge of the bounding box
 */
class BoundBox(val xmin: Float, val xmax: Float, val ymin: Float, val ymax: Float){
    /**
     * Method returning RectF representation of the BoundBox, with absolute coordinates calculated
     * with use of provided width and height.
     *
     * @param width width of the image to which normalized coordinates should be adjusted
     * @param height height of the image to which normalized coordinates should be adjusted
     */
    fun getRectF(width: Int, height: Int): RectF{
        return RectF(xmin * width, ymax * height, xmax * width, ymin * height)
    }
}