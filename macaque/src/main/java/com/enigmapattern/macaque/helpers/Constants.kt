package com.enigmapattern.macaque.helpers

class Constants {
    companion object {
        const val pixelSize = 3
    }
}

enum class InputNormalization{
    NONE,
    DIVIDED_BY_128,
    DIVIDED_BY_255
}
