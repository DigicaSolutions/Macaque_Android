package com.enigmapattern.macaque.dataproviders.input

import com.enigmapattern.macaque.dataproviders.DataNotReadyException
import java.nio.ByteBuffer

/**
 * Interface for Input Data Provider
 *
 * Objects implementing this interface are responsible for taking data from user and translating it
 * to form accepted by model. There are available few implementations of
 * InputDataProvider (e.g. BitmapToUINT8DataProvider, BitmapToFloat32DataProvider), more can be
 * easily implemented.
 */
abstract class InputDataProvider {
    /**
     * Method which takes data provided by the user to Input Data Provider and returns it to
     * Predictor in form accepted by the model.
     *
     * @param forShapes array of shapes used in process of generating data for Predictor
     * @return Array<out Any> array of objects, usually Byte Buffers used later as input for model.
     * If single input is expected by the model then single element array should be returned.
     * @throws DataNotReadyException if the data stored in Input Data Provider is not ready to be
     * used in prediction or there is no data provided by the user yet.
     */
    @Throws(DataNotReadyException::class)
    protected abstract fun data(forShapes: Array<IntArray>): Array<out Any>

    /**
     * Helper method for making 'data' method available within module.
     *
     * @param forShapes array of shapes used in process of generating data for Predictor
     * @return Array<out Any> array of objects, usually Byte Buffers used later as input for model.
     * If single input is expected by the model then single element array should be returned.
     * @throws DataNotReadyException if the data stored in Input Data Provider is not ready to be
     * used in prediction or there is no data provided by the user yet.
     */
    internal fun _data(forShapes: Array<IntArray>): Array<out Any>{
        return data(forShapes)
    }

    /**
     * Method for clearing all resources taken by Input Data Provider. After running it Input Data
     * Provider should not be used.
     */
    protected abstract fun close()

    /**
     * Helper method for making 'close' method available within module.
     */
    internal fun _close(){
        close()
    }
}
