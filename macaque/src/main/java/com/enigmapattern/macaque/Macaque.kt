package com.enigmapattern.macaque

import com.enigmapattern.macaque.dataproviders.input.InputDataProvider
import com.enigmapattern.macaque.dataproviders.output.OutputDataProvider
import com.enigmapattern.macaque.predictors.PredictorBase

/**
 * Main class of the library
 *
 * This is the main class of the library. It is used for making predictions based on data read from
 * Input Data Provider, with use of Predictor and storing results in Output Data Provider. It
 * utilizes these three submodules:
 * * Input Data Provider - responsible for taking data from user and translating it to form accepted
 * by the TensorFlow Lite Interpreter. It is object of class implementing InputDataProvider interface.
 * Reference to this object has to be provided during initialization of Macaque. There are
 * available few implementations of InputDataProvider (e.g. BitmapToUINT8DataProvider,
 * BitmapToFloat32DataProvider), more can be easily implemented.
 * * Predictor - makes prediction based on provided model. It is object of class implementing
 * PredictorBase. Reference to this object has to be provided during initialization
 * of Macaque. There are available few implementations of Predictor (TensorFlow and TensorFlow Lite),
 * more can be easily implemented.
 * * Output Data Provider - responsible for gathering output of predictions and returning it to
 * the user either in raw form or parsed to expected form. It is object of class extending
 * OutputDataProvider class. Reference to this object has to be provided during initialization
 * of Macaque. There are available few implementations of OutputDataProvider (e.g.
 * Detector4outputsBoundingBoxesDataProvider, ClassifierDataProvider), more can be easily implemented.
 *
 * Sample usage:
 * 1. Select and instantiate proper input data provider and results provider, eg.:
 * ```
 * val inputDataProvider = = BitmapToUINT8DataProvider()
 * val outputDataProvider = Detector4outputsBoundingBoxesDataProvider()
 * ```
 * 2. Instantiate predictor object with at least model path and optional flags for GPU and NNApi
 * usage (TensorFlow Lite predictor):
 * ```
 * val modelPath = File(modelsDir, "detect.tflite")
 * val predictor = TensorFlowLitePredictor(modelPath.absolutePath, useGPU = true)
 * ```
 * 3. Initialize Macaque:
 * ```
 * val macaque = Macaque(inputDataProvider, outputDataProvider, predictor)
 * ```
 * 4. Provide input data to input data provider:
 * ```
 * inputDataProvider.currentImage = someBitmap
 * ```
 * 5. Run prediction:
 * ```
 * macaque.predict()
 * ```
 * 6. Get parsed results from results provider:
 * ```
 * val resultsList = outputDataProvider.getResults()
 * ```
 *
 * @param T the type of InputDataProvider
 * @param U the type of OutputDataProvider
 * @param V the type of PredictorBase
 * @param inputDataProvider instance of object implementing InputDataProvider interface
 * @param outputDataProvider instance of object extending OutputDataProvider class
 * @param predictor instance of object extending PredictorBase class
 * @property predictionInProgress boolean value indicating whether prediction is currently in progress
 * @constructor Private constructor, base for public constructors
 */
class Macaque<T: InputDataProvider, U: OutputDataProvider, V: PredictorBase> constructor(private val inputDataProvider: T,
                                                                                         private val outputDataProvider: U,
                                                                                         private val predictor: V) {

    init {
        predictor._initialize(outputDataProvider)
    }

    var predictionInProgress = false
        private set

    /**
     * Method for performing prediction
     *
     * Input data is taken from InputDataProvider. Then prediction method is ran on instance
     * of TensorFlow Lite interpreter and raw results are stored in OutputDataProvider.
     *
     * The copy of data taken from InputDataProvider is then cleared in order to avoid memory leaks.
     *
     * This operation, depending on model, might be very expensive. It is highly recommended to 
     * perform this operation in background thread.
     *
     * @throws PredictionInProgressException when method is ran while there is already prediction
     * in progress.
     */
    fun predict(){
        if(predictionInProgress){
            throw PredictionInProgressException("Prediction already in progress")
        }

        predictionInProgress = true
        predictor._predict(inputDataProvider, outputDataProvider)
        predictionInProgress = false

    }

    /**
     * Method for clearing all resources taken by Predictor and its submodules. It should be ran
     * when Predictor is no longer needed as there is no possibility to run predictions after
     * closing Predictor.
     *
     * @throws PredictionInProgressException if prediction is in progress when this method is ran.
     */
    fun close(){
        if(predictionInProgress){
            throw PredictionInProgressException("Prediction is in progress")
        }
        predictor._close()
        inputDataProvider._close()
        outputDataProvider._close()
    }

}

/**
 * Exception thrown if there is prediction in progress and action is taken that can interrupt it.
 */
class PredictionInProgressException(msg: String = "Prediction is in progress"): Exception(msg)
