package com.enigmapattern.macaque.results

/**
 * Class representing single result of classification.
 *
 * Representation of single object classified by classification model.
 *
 * @property cls class of classified object
 * @property score confidence with which object was classified
 */
class ClassificationResult (val cls: Int, val score: Float){
}
