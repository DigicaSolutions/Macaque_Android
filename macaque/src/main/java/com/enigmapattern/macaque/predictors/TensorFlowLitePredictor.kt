package com.enigmapattern.macaque.predictors

import com.enigmapattern.macaque.dataproviders.input.InputDataProvider
import com.enigmapattern.macaque.dataproviders.output.OutputDataProvider
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer

/**
 * Class for TensorFlow Lite Predictor
 *
 * Predictor class for TensorFlow Lite models.
 *
 * @constructor private constructor, that should not be instantiated alone, as base for other
 * constructors
 */
class TensorFlowLitePredictor private constructor(): PredictorBase(){

    private lateinit var interpreter: Interpreter
    private lateinit var inputTensorsShapes: Array<IntArray>
    private lateinit var outputTensorsShapes: Array<IntArray>
    private lateinit var inputTensorsTypes: Array<DataType>
    private lateinit var outputTensorsTypes: Array<DataType>
    /**
     * Constructor of the predictor that takes as argument path to model. Optionally it takes flags
     * indicating whether GPU or NNApi should be used during inference - both set to false by
     * default. Please note that setting useGPU to true will disregard value of useNNApi.
     *
     * Internally it initializes TensorFlow Lite Interpreter.
     *
     * @param modelPath path to model
     * @param useGPU flag to indicate whether GPU delegate should be used
     * @param useNNApi flag to indicate whether NNApi should be used
     * @constructor constructor for Predictor
     */
    constructor(modelPath: String, useGPU: Boolean = false, useNNApi: Boolean = false): this(){
        val modelFile = File(modelPath)
        if (!modelFile.exists()){
            throw ModelFileNotFoundException("There is no model file on provided model path")
        }
        val options = Interpreter.Options()
        if (useGPU) {
            val delegate = GpuDelegate()
            options.addDelegate(delegate)
        } else if (useNNApi) {
            val delegate = NnApiDelegate()
            options.addDelegate(delegate)
        }
        interpreter = Interpreter(modelFile, options)
    }

    /**
     * Constructor of the predictor that takes as arguments path to model and TensorFlow Lite
     * Interpreter Options. Optionally it takes flags indicating whether GPU or NNApi should be used
     * during inference - both set to false by default. Please note that setting useGPU to true will
     * disregard value of useNNApi.
     *
     * Internally it initializes TensorFlow Lite Interpreter.
     *
     * @param modelPath path to model
     * @param options instance of TensorFlow Lite Interpreter Options object
     * @constructor constructor for Predictor
     */
    constructor(modelPath: String, options: Interpreter.Options): this(){
        val modelFile = File(modelPath)
        if (!modelFile.exists()){
            throw ModelFileNotFoundException("There is no model file on provided model path")
        }
        interpreter = Interpreter(modelFile, options)
    }

    /**
     * Constructor of the predictor that takes as parameter reference to TensorFlow Lite
     * Interpreter.
     *
     * Internally it finishes initialization of TensorFlow Lite Interpreter.
     *
     * @param tf instance of TensorFlow Lite Interpreter
     * @constructor secondary constructor of the predictor in case when TensorFlow Lite Interpreter
     * is created somewhere else (e.g. in order to customize it)
     */
    constructor(tf: Interpreter): this(){
        interpreter = tf
    }

    /**
     * This method finishes initialization of predictor and Output Data Provider object .
     *
     * @param outputDataProvider instance of Output Data Provider
     */
    override fun initialize(outputDataProvider: OutputDataProvider){
        val inputTensors = Array(interpreter.inputTensorCount){ i -> interpreter.getInputTensor(i) }
        val outputTensors = Array(interpreter.outputTensorCount){ i -> interpreter.getOutputTensor(i) }

        inputTensorsShapes = Array(interpreter.inputTensorCount){ i -> inputTensors[i].shape()}
        inputTensorsTypes = Array(interpreter.inputTensorCount){ i -> inputTensors[i].dataType()}

        outputTensorsShapes = Array(interpreter.outputTensorCount) { i -> outputTensors[i].shape() }
        outputTensorsTypes = Array(interpreter.outputTensorCount){ i -> outputTensors[i].dataType()}

        val outputTensorsTypeSizes = Array(outputTensorsTypes.size){ i -> outputTensorsTypes[i].byteSize() }
        outputDataProvider._initializeOutputsHashMap(outputTensorsTypeSizes, outputTensorsShapes)
    }

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
     * @param inputDataProvider instance of Input Data Provider
     * @param outputDataProvider instance of Output Data Provider
     */
    override fun predict(inputDataProvider: InputDataProvider, outputDataProvider: OutputDataProvider) {
        val inputData = inputDataProvider._data(inputTensorsShapes) as Array<ByteBuffer>
        outputDataProvider._resetOutputsHashMap()
        interpreter.runForMultipleInputsOutputs(inputData, outputDataProvider.outputsHashMap!!)
        for (buff in inputData) buff.clear()

    }

    /**
    * Method for clearing all resources taken by TensorFlow Lite Predictor.
    */
    override fun close() {
        interpreter.close()
    }

}