package com.enigmapattern.macaque.dataproviders

/**
 * Exception thrown when data is not ready yet to be used.
 */
class DataNotReadyException(msg: String): Exception(msg)
class UnexpectedDataShapeException(msg: String): Exception(msg)