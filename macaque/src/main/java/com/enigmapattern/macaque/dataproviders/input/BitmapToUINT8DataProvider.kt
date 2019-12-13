package com.enigmapattern.macaque.dataproviders.input

import com.enigmapattern.macaque.helpers.InputNormalization
import org.tensorflow.lite.DataType

/**
 * Input Data Provider converting Bitmaps to UINT8 Data
 *
 * Object of this class can be used Input Data Provider. It takes Bitmap as input from user and
 * provides single output of UINT8 Data to the Predictor.
 *
 * @param inputNormalization object of type InputNormalization, indicating how data should be
 * normalized before putting into returned ByteBuffer.
 * @property dataType DataType field, filled by classes extending this class, which informs what
 * DataType of RGB Data is expected by the predictor.
 */
class BitmapToUINT8DataProvider(inputNormalization: InputNormalization = InputNormalization.NONE): BitmapToRGBDataProviderBase(inputNormalization) {
    override val dataType = DataType.UINT8
}