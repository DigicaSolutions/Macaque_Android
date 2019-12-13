package com.enigmapattern.macaque.predictors

import com.enigmapattern.macaque.dataproviders.input.InputDataProvider
import com.enigmapattern.macaque.dataproviders.output.OutputDataProvider

/**
 * Abstract base class for Predictors
 *
 * Objects extending this class are responsible for performing predictions on data taken from
 * InputDataProvider and storing results in OutputDataProvider. There are available two
 * implementations of PredictorBase at this moment (TensorFlowPredictor and TensorFlowLitePredictor),
 * more will be added in the future or can be easily implemented by the user.
 */
abstract class PredictorBase{

    /**
     * This method finishes initialization of predictor and Output Data Provider object .
     *
     * @param outputDataProvider instance of Output Data Provider
     */
    protected abstract fun initialize(outputDataProvider: OutputDataProvider)

    /**
     * Helper method for making 'initialize' method available within module.
     *
     * @param outputDataProvider instance of Output Data Provider
     */
    internal fun _initialize(outputDataProvider: OutputDataProvider){
        initialize(outputDataProvider)
    }

    /**
     * Method for performing prediction
     *
     * Input data is taken from InputDataProvider. Then prediction method is ran on instance
     * of Interpreter and raw results are stored in OutputDataProvider.
     *
     * This operation, depending on model, might be very expensive. It is highly recommended to
     * perform this operation in background thread.
     * @param inputDataProvider reference to Input Data Provider object
     * @param outputDataProvider reference to Output Data Provider object
     */
    protected abstract fun predict(inputDataProvider: InputDataProvider, outputDataProvider: OutputDataProvider)

    /**
     * Helper method for making 'predict' method available within module.
     * @param inputDataProvider reference to Input Data Provider object
     * @param outputDataProvider reference to Output Data Provider object
     */
    internal fun _predict(inputDataProvider: InputDataProvider, outputDataProvider: OutputDataProvider){
        predict(inputDataProvider, outputDataProvider)
    }

    /**
     * Method for clearing all resources taken by Predictor.
     */
    protected abstract fun close()

    /**
     * Helper method for making 'close' method available within module.
     */
    internal fun _close(){
        close()
    }
}