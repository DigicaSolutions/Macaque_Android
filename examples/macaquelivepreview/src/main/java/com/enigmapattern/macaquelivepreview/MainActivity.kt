package com.enigmapattern.macaquelivepreview

import android.content.pm.PackageManager
import android.graphics.*
import android.os.*
import android.util.DisplayMetrics
import android.util.Rational
import androidx.appcompat.app.AppCompatActivity
import android.util.Size
import android.widget.Toast
import androidx.camera.core.*
import com.enigmapattern.macaque.results.DetectionResult
import com.enigmapattern.macaque.Macaque
import com.enigmapattern.macaque.dataproviders.input.BitmapToUINT8DataProvider
import com.enigmapattern.macaque.dataproviders.output.Detector4outputsBoundingBoxesDataProvider
import com.enigmapattern.macaque.helpers.toBitmap
import com.enigmapattern.macaque.predictors.TensorFlowLitePredictor
import kotlinx.android.synthetic.main.activity_main.*

import java.io.File
import java.lang.Math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var macaque: Macaque<BitmapToUINT8DataProvider,
            Detector4outputsBoundingBoxesDataProvider,
            TensorFlowLitePredictor>

    private lateinit var inputDataProvider: BitmapToUINT8DataProvider
    private lateinit var outputDataProvider: Detector4outputsBoundingBoxesDataProvider

    private lateinit var resultsBitmap: Bitmap
    private lateinit var resultsCanvas: Canvas

    private var lensFacing = CameraX.LensFacing.BACK


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        // Find absolute path to model file
        val modelsDir = File(getExternalFilesDir(null), "models")
        val modelPath = File(modelsDir, "detect.tflite").absolutePath

        // Instantiate Input Data Provider and Output Data Provider. In this case we use data
        // providers available out of the box.
        inputDataProvider = BitmapToUINT8DataProvider()
        outputDataProvider = Detector4outputsBoundingBoxesDataProvider()
        val tf = TensorFlowLitePredictor(modelPath)
        // Initialize predictor with previously defined variables
        macaque = Macaque(inputDataProvider, outputDataProvider, tf)

        if (checkSelfPermission("android.permission.CAMERA") == PackageManager.PERMISSION_GRANTED) {
            imageView.post { startCamera() }
            imageView.post { setUpCanvas() }
        } else {

            requestPermissions(arrayOf("android.permission.CAMERA"), 2);
        }

    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>,
                                            grantResults: IntArray) {

        if (requestCode == 2) {
            if (checkSelfPermission("android.permission.CAMERA") == PackageManager.PERMISSION_GRANTED) {
                startCamera()
                setUpCanvas()
            } else {
                Toast.makeText(this, "Permission to use Camera is required to use this app.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }

    /**
     * Method that initializes and starts usage of camera, both to preview as well as to inference.
     * It utilizes cameraX api in version 1.0.0-alpha05. Please note that cameraX api is not stable
     * and probably will change.
     */
    private fun startCamera() {
        val metrics = DisplayMetrics()
        imageView.display.getRealMetrics(metrics)

        val screenSize = Size(imageView.width, imageView.height)

        val screenAspectRatio = metrics.heightPixels * 1.0f / metrics.widthPixels
        val aspectRatio =
            if (abs(screenAspectRatio - 16.0/9) - abs(screenAspectRatio - 4.0/3) > 0) {
                Rational(3, 4)
            } else {
                Rational(9, 16)
            }

        val previewConfig = PreviewConfig.Builder().apply {
            setLensFacing(lensFacing)
            setTargetResolution(screenSize)
            setTargetAspectRatio(aspectRatio)
            setTargetRotation(windowManager.defaultDisplay.rotation)
            setTargetRotation(imageView.display.rotation)
        }.build()

        val preview = Preview(previewConfig)
        preview.setOnPreviewOutputUpdateListener {
            imageView.surfaceTexture = it.surfaceTexture
        }


        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            // Analyzer with which all predictions will be performed is ran on separate thread
            // so that we do not block UI thread.
            val analyzerThread = HandlerThread("InferenceThread").apply {
                start()
            }
            setTargetAspectRatio(aspectRatio)
            setCallbackHandler(Handler(analyzerThread.looper))
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
        }.build()

        val analyzerUseCase = ImageAnalysis(analyzerConfig)
        // Definition of analyzer for live inference
        analyzerUseCase.analyzer =
            ImageAnalysis.Analyzer { imageProxy: ImageProxy, rotationDegrees: Int ->

                // Take reference to image out of imageProxy
                val image = imageProxy.image!!
                // Use convenience extension method provided with Macaque library to convert
                // Image to Bitmap
                val bitmap = image.toBitmap()

                // Create appropriately rotated Bitmap
                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)
                val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width,
                    bitmap.height, rotateMatrix, true)
                bitmap.recycle()

                // Provide prepared Bitmap to Input Data Provider
                inputDataProvider.currentImage = rotatedBitmap
                // Perform prediction
                macaque.predict()
                // Get parsed results from Output Data Provider
                val resultsList = outputDataProvider.getResults()
                // Use results to draw bound boxes
                drawOnRectProjectedBitMap(resultsList)
            }
        CameraX.bindToLifecycle(this, preview, analyzerUseCase)
    }

    /**
     * Helper method for setting up canvas for drawing inference results. It has to be ran after
     * imageView is fully inflated but before inference is ran.
     */
    private fun setUpCanvas() {
        resultsBitmap = Bitmap.createBitmap(imageView.width, imageView.height, Bitmap.Config.ARGB_8888)
        resultsCanvas = Canvas(resultsBitmap)
        results.setImageBitmap(resultsBitmap)
    }

    /**
     * Method for drawing results of object detection.
     */
    private fun drawOnRectProjectedBitMap(resultsList: List<DetectionResult>) {

        resultsCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.color = Color.WHITE
        paint.strokeWidth = 5f
        for (detectionResult in resultsList) {
            if (detectionResult.score < 0.55f) {
                continue
            }
            resultsCanvas.drawRect(detectionResult.bbox.getRectF(imageView.width, imageView.height), paint)
        }
        results.invalidate()
    }
}
