package com.enigmapattern.macaque.dataproviders.input

import android.graphics.Bitmap
import com.enigmapattern.macaque.dataproviders.DataNotReadyException
import com.enigmapattern.macaque.helpers.InputNormalization
import com.enigmapattern.macaque.helpers.convertBitmapToByteBuffer
import com.enigmapattern.macaque.helpers.resize
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer

/**
 * Base class for Input Data Provider converting Bitmaps to RGB Data
 *
 * This class is used as base for Input Data Providers which take Bitmap as input from user and
 * provide single output of RGB Data to the Predictor. It implements InputDataProvider interface
 * but cannot be used directly as Input Data Provider.
 *
 * @param inputNormalization object of type InputNormalization, indicating how data should be
 * normalized before putting into returned ByteBuffer.
 * @property currentImage nullable field where user provides Bitmap to be processed
 * @property dataType DataType field, filled by classes extending this class, which informs what
 * DataType of RGB Data is expected by the predictor.
 */
abstract class BitmapToRGBDataProviderBase(private val inputNormalization: InputNormalization):
    InputDataProvider() {

    var currentImage: Bitmap? = null

    abstract val dataType: DataType

    /**
     * Method which takes data stored by the user in the currentImage field and returns it to
     * Predictor in form accepted by the TensorFlow Lite Interpreter.
     *
     * @param forShapes array of shapes used in process of generating data for Predictor
     * @return Array<ByteBuffer> array of byte buffers used later as input for TensorFlow Lite
     * Interpreter. If single input is expected by the model then single element array should be
     * returned.
     * @throws DataNotReadyException if the there is no Bitmap stored in currentImage field.
     */
    override fun data(forShapes: Array<IntArray>): Array<ByteBuffer> {
        val inputLayerShape = forShapes[0]
        val inputWidth = inputLayerShape[1]
        val inputHeight = inputLayerShape[2]
        if (currentImage != null) {
            val resizedBitmap = currentImage!!.resize(inputWidth, inputHeight)
            val result = arrayOf(
                convertBitmapToByteBuffer(resizedBitmap, dataType, inputNormalization)
            )
            resizedBitmap.recycle()
            return result
        } else {
            throw DataNotReadyException(
                "There was no image provided yet."
            )
        }
    }

    /**
     * Method for clearing all resources taken by Input Data Provider. Since this Input Data
     * Provider does not own any resources, this method has empty body.
     */
    override fun close(){
        currentImage = null
    }

}

