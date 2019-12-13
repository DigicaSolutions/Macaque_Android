package com.enigmapattern.macaque.dataproviders.output

import com.enigmapattern.macaque.dataproviders.DataNotReadyException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Abstract base class for Output Data Provider
 *
 * Objects extending this class are responsible for gathering output of predictions and returning it
 * to the user either in raw form or parsed to expected form. There are available few
 * implementations of OutputDataProvider (e.g. Detector4outputsBoundingBoxesDataProvider,
 * ClassifierDataProvider), more can be easily implemented.
 * @property outputsHashMap nullable field containing HashMap to which results of inference
 * performed by model will be loaded. Before inference is done, HashMap has to be initialized (which
 * is done by default with initializeOutputHashMap method ran by the Predictor).
 * @property outputTensorsShapes nullable helper field containing array of shapes of output tensors
 * of loaded model.
 */
abstract class OutputDataProvider {

    var outputsHashMap: HashMap<Int, Any>? = null
        get() {
            if(field != null) {
                if (field!!.size != 0) {
                    return field
                } else {
                    throw DataNotReadyException("Outputs Hashmap has no buffers or arrays assigned")
                }
            } else {
                throw DataNotReadyException("Outputs Hashmap is null")
            }
        }
        protected set

    protected var outputTensorsShapes: Array<IntArray>? = null

    /**
     * Method for parsing and returning results.
     *
     * It operates on outputsHashMap and with use of outputTensorsShapes parses it and provides list
     * of results in form expected by the user.
     * @return List of results (if single result is returned it should be returned as list with
     * single element)
     */
    abstract fun getResults(): List<Any>

    /**
     * Default method for initializing outputsHashMap. For each output tensor it creates appropriate
     * byte buffer and assigns it to proper index of outputsHashMap.
     *
     * This method can be overridden to create arrays instead of ByteBuffers which makes it easier
     * to parse later but has to be done separately for each different model (due to difference of
     * shapes and output DataType).
     * If you override this method, `resetOutputsHashMap` and `close` methods should also be
     * overridden.
     *
     * @param outputTensorsTypeSizes array of byte sizes of output Tensors of loaded model
     * @param outputTensorsShapes array of shapes of output Tensors of loaded model
    */
    protected fun initializeOutputsHashMap(outputTensorsTypeSizes: Array<Int>, outputTensorsShapes: Array<IntArray>){
        val outputsHashMapTmp = HashMap<Int, Any>()
        for (i in outputTensorsTypeSizes.indices){
            val shape = outputTensorsShapes[i]
            var bufferSize = outputTensorsTypeSizes[i]
            for (dim in shape){
                bufferSize *= dim
            }
            val buffer = ByteBuffer.allocateDirect(bufferSize)
            buffer.order(ByteOrder.nativeOrder())
            buffer.rewind()
            outputsHashMapTmp[i] = buffer
        }
        outputsHashMap = outputsHashMapTmp
        this.outputTensorsShapes = outputTensorsShapes

    }

    /**
     * Helper method for making 'initializeOutputsHashMap' method available within module.
     *
     * @param outputTensorsTypeSizes array of byte sizes of output Tensors of loaded model
     * @param outputTensorsShapes array of shapes of output Tensors of loaded model
     */
    internal fun _initializeOutputsHashMap(outputTensorsTypeSizes: Array<Int>, outputTensorsShapes: Array<IntArray>) {
        initializeOutputsHashMap(outputTensorsTypeSizes, outputTensorsShapes)
    }

    /**
     * Method rewinding ByteBuffers in outputsHashMap.
     *
     * If you have overridden initializeOutputHashMap method to utilize arrays, you have to override
     * this method too (e.g. give it empty body).
     */
    protected fun resetOutputsHashMap(){
        with (outputsHashMap as HashMap<Int, ByteBuffer>){
            for (buffer in this.values){
                buffer.rewind()
            }
        }
    }

    /**
     * Helper method for making 'resetOutputHashMap' method available within module.
     */
    internal fun _resetOutputsHashMap(){
        resetOutputsHashMap()
    }

    /**
     * Method for clearing all resources taken by Output Data Provider.
     *
     * If you have overridden initializeOutputHashMap method to utilize arrays, you have to override
     * this method too.
     */
    protected fun close() {
        if (outputsHashMap != null) {
            with (outputsHashMap as HashMap<Int, ByteBuffer>){
                for (buffer in this.values){
                    buffer.clear()
                }
            }
            outputsHashMap!!.clear()
            outputsHashMap = null
        }
    }

    /**
     * Helper method for making 'close' method available within module.
     */
    internal fun _close(){
        close()
    }
}