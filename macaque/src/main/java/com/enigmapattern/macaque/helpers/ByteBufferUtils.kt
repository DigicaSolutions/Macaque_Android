package com.enigmapattern.macaque.helpers

import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Helper method returning ByteBuffer converted from Bitmap object. Conversion is based on values of
 * provided DataType and InputNormalization enums. Usually if model is quantized, DataType is UINT8
 * and InputNormalization is NONE, while for non-quantized model usual values are respectively
 * FLOAT32 and DIVIDED_BY_128.
 *
 * @param bitmap source Bitmap object
 * @param inputType object of type DataType, indicating type of data that should be put into returned
 * ByteBuffer
 * @param inputNormalization object of type InputNormalization, indicating how data should be
 * normalized before putting into returned ByteBuffer
 * @return ByteBuffer object filled with data from input bitmap
 * @throws DataTypeNotSupportedException if not supported inputType parameter is provided
 */
fun convertBitmapToByteBuffer(bitmap: Bitmap, inputType: DataType, inputNormalization: InputNormalization): ByteBuffer {
    val numBytesPerChannel = when (inputType) {
        DataType.FLOAT32 -> 4
        DataType.UINT8 -> 1
        DataType.INT64 -> throw DataTypeNotSupportedException("INT64 input tensors not supported at this time")
        DataType.INT32 -> throw DataTypeNotSupportedException("INT32 input tensors not supported at this time")
        DataType.STRING -> throw DataTypeNotSupportedException("String tensors not supported at this time")
    }

    val result =
        ByteBuffer.allocateDirect(bitmap.width * bitmap.height * Constants.pixelSize * numBytesPerChannel)
    result.order(ByteOrder.nativeOrder())

    val intValues = IntArray(bitmap.width * bitmap.height)

    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    for (i in 0 until bitmap.width) {
        for (j in 0 until bitmap.height) {
            val pixelValue = intValues[i * bitmap.height + j]
            when (inputType) {
                DataType.UINT8 -> {
                    // Quantized model
                    result.put((pixelValue shr 16 and 0xFF).toByte())
                    result.put((pixelValue shr 8 and 0xFF).toByte())
                    result.put((pixelValue and 0xFF).toByte())
                }
                DataType.FLOAT32 -> {
                    when (inputNormalization) {
                        InputNormalization.DIVIDED_BY_128 -> { // Float model
                            result.putFloat(((pixelValue shr 16 and 0xFF) / 128f) - 1)
                            result.putFloat(((pixelValue shr 8 and 0xFF) / 128f) - 1)
                            result.putFloat(((pixelValue and 0xFF) / 128f) - 1)
                        }
                        else -> {
                            result.putFloat((pixelValue shr 16 and 0xFF) / 255f)
                            result.putFloat((pixelValue shr 8 and 0xFF) / 255f)
                            result.putFloat((pixelValue and 0xFF) / 255f)
                        }
                    }
                }
                DataType.INT64 -> throw DataTypeNotSupportedException("INT64 input tensors not supported at this time")
                DataType.INT32 -> throw DataTypeNotSupportedException("INT32 input tensors not supported at this time")
                DataType.STRING -> throw DataTypeNotSupportedException("String tensors not supported at this time")
            }
        }
    }
    result.rewind()

    return result
}

/**
 * Exception thrown when provided DataType is not supported.
 */
class DataTypeNotSupportedException(msg: String): Exception(msg)