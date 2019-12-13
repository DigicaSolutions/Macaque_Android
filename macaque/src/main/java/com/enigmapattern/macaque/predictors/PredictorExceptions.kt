package com.enigmapattern.macaque.predictors

/**
 * Exception thrown when model file is not found.
 */
class ModelFileNotFoundException(msg: String): Exception(msg)

/**
 * Exception thrown when there is no data type provided or data type is not supported.
 */
class DataTypeNotSupportedException(msg: String): Exception(msg)