package com.enigmapattern.macaque.predictors

import com.enigmapattern.macaque.dataproviders.input.InputDataProvider
import com.enigmapattern.macaque.dataproviders.output.OutputDataProvider
import org.tensorflow.DataType
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer

/**
 * Class for TensorFlow Predictor
 *
 * Predictor class for TensorFlow models.
 *
 * @param inputTensorsNames array of names of input tensors
 * @param inputTensorsShapes array of shapes of input tensors
 * @param outputTensorsNames array of names of output tensors
 * @param outputTensorsShapes array of shapes of output tensors
 * @constructor private constructor, that should not be instantiated alone, as base for other
 * constructors
 */
class TensorFlowPredictor private constructor(private val inputTensorsNames: Array<String>,
                                              private val inputTensorsShapes: Array<IntArray>,
                                              private val outputTensorsNames: Array<String>,
                                              private val outputTensorsShapes: Array<IntArray>): PredictorBase(){

    private lateinit var interpreter: TensorFlowInferenceInterface
    private lateinit var inputTensorsTypes: Array<DataType>
    private lateinit var outputTensorsTypes: Array<DataType>

    /**
     * Constructor of the predictor that takes as argument path to model.
     *
     * Internally it initializes TensorFlow Inference Interface.
     *
     * @param modelPath path to model
     * @param inputTensorsNames array of names of input tensors
     * @param inputTensorsShapes array of shapes of input tensors
     * @param outputTensorsNames array of names of output tensors
     * @param outputTensorsShapes array of shapes of output tensors
     * @constructor constructor for Predictor
     */
    constructor(modelPath: String, inputTensorsNames: Array<String>, inputTensorsShapes: Array<IntArray>,
                outputTensorsNames: Array<String>, outputTensorsShapes: Array<IntArray>):
            this(inputTensorsNames, inputTensorsShapes, outputTensorsNames, outputTensorsShapes){

        val modelFile = File(modelPath)
        if (!modelFile.exists()){
            throw ModelFileNotFoundException("There is no model file on provided model path")
        }
        val inputStream = FileInputStream(modelFile)
        interpreter = TensorFlowInferenceInterface(inputStream)
        inputStream.close()

    }

    /**
     * Constructor of the predictor that takes as parameter reference to TensorFlow Inference
     * Interface.
     *
     * Internally it finishes initialization of TensorFlow Inference Interface.
     *
     * @param interpreter instance of TensorFlow Inference Interface
     * @param inputTensorsNames array of names of input tensors
     * @param inputTensorsShapes array of shapes of input tensors
     * @param outputTensorsNames array of names of output tensors
     * @param outputTensorsShapes array of shapes of output tensors
     * @constructor secondary constructor of the predictor in case when TensorFlow Inference Interface
     * is created somewhere else (e.g. in order to customize it)
     */
    constructor(interpreter: TensorFlowInferenceInterface, inputTensorsNames: Array<String>,
                inputTensorsShapes: Array<IntArray>, outputTensorsNames: Array<String>,
                outputTensorsShapes: Array<IntArray>): this(inputTensorsNames, inputTensorsShapes,
                outputTensorsNames, outputTensorsShapes){

        this.interpreter = interpreter
    }

    /**
     * This method finishes initialization of predictor and Output Data Provider object.
     *
     * @param outputDataProvider instance of Output Data Provider
     */
    override fun initialize(outputDataProvider: OutputDataProvider){
        inputTensorsTypes = Array(inputTensorsNames.size){ i -> interpreter.graph().operation(inputTensorsNames[i]).output<Float>(0).dataType() }
        outputTensorsTypes = Array(outputTensorsNames.size){ i -> interpreter.graph().operation(outputTensorsNames[i]).output<Float>(0).dataType() }

        val outputTensorsTypeSizes = Array(outputTensorsTypes.size){ i -> outputTensorsTypes[i].byteSize() }
        outputDataProvider._initializeOutputsHashMap(outputTensorsTypeSizes, outputTensorsShapes)
    }

    /**
     * Method for performing prediction
     *
     * Input data is taken from InputDataProvider. Then prediction method is ran on instance
     * of TensorFlow Inference Interface and raw results are stored in OutputDataProvider.
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

        for (i in inputData.indices){
            val shapeInLongs = Array(inputTensorsShapes[i].size){
                    j -> inputTensorsShapes[i][j].toLong()
            }.toLongArray()
            interpreter.feed(inputTensorsNames[i], inputData[i], *shapeInLongs)
        }

        interpreter.run(outputTensorsNames, false)

        for (i in outputTensorsNames.indices){
            interpreter.fetch(outputTensorsNames[i], outputDataProvider.outputsHashMap!![i] as ByteBuffer)
        }

        for (buff in inputData) buff.clear()

    }

    /**
    * Method for clearing all resources taken by TensorFlow Lite Predictor.
    */
    override fun close() {
        interpreter.close()
    }

}