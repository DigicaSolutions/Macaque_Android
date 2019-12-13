package com.enigmapattern.macaque.helpers

import android.graphics.Bitmap

/**
 * Extension method returning Bitmap object created by resizing of source bitmap.
 */
fun Bitmap.resize(toWidth: Int, toHeight: Int): Bitmap {
    return Bitmap.createScaledBitmap(this, toWidth, toHeight, false)
}


