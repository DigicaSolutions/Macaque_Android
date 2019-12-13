package com.enigmapattern.macaquepreview

import android.Manifest
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.*
import androidx.appcompat.app.AppCompatActivity
import android.provider.MediaStore
import androidx.core.content.FileProvider
import com.enigmapattern.macaque.Macaque
import com.enigmapattern.macaque.dataproviders.input.BitmapToFloat32DataProvider
import com.enigmapattern.macaque.dataproviders.output.ClassifierDataProvider
import com.enigmapattern.macaque.predictors.TensorFlowLitePredictor

import kotlinx.android.synthetic.main.activity_main.*

import java.io.File
import java.io.IOException
import java.lang.Exception
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {


    private val RESULT_LOAD_IMAGE = 2

    private lateinit var macaque: Macaque<BitmapToFloat32DataProvider,
            ClassifierDataProvider, TensorFlowLitePredictor>

    private lateinit var pathToFile: String

    private lateinit var inputDataProvider: BitmapToFloat32DataProvider
    private lateinit var outputDataProvider: ClassifierDataProvider

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestPermissions(
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ), 2
        )

        btnTakePic.setOnClickListener { dispatchPictureTakerAction() }
        btnLoadPic.setOnClickListener { loadPictureAction() }
        btnMakePred.setOnClickListener { onDetectClicked() }

        val modelsDir = File(getExternalFilesDir(null), "models")
        val modelPath = File(modelsDir, "mobilenet_v1_1.0_224.tflite").absolutePath
        inputDataProvider = BitmapToFloat32DataProvider()
        outputDataProvider = ClassifierDataProvider()
        val tf = TensorFlowLitePredictor(modelPath)
        macaque = Macaque(inputDataProvider, outputDataProvider, tf)
    }

    private fun onDetectClicked() {
        val bitmap = BitmapFactory.decodeFile(pathToFile)
        inputDataProvider.currentImage = bitmap

        try{
            macaque.predict()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        val resultsList = outputDataProvider.getResults()

        result.text = "Classification: ${resultsList[0].cls}"

    }

    private fun dispatchPictureTakerAction() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                val photoFile: File? = createImageFile()

                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.enigmapattern.macaquepreview.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, 1)
                }
            }
        }

    }

    private fun loadPictureAction() {
        val i = Intent(
            Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        )

        startActivityForResult(i, RESULT_LOAD_IMAGE)
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File = getExternalFilesDir(Environment.DIRECTORY_PICTURES)!!
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        ).apply {
            pathToFile = absolutePath
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == 1) {
            val bitmap = BitmapFactory.decodeFile(pathToFile)
            imageView.setImageBitmap(bitmap)

        } else if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            val selectedImage = data.data
            val filePathColumn = arrayOf(MediaStore.Images.Media.DATA)

            val cursor = contentResolver.query(
                selectedImage!!,
                filePathColumn, null, null, null
            )
            cursor!!.moveToFirst()

            val columnIndex = cursor.getColumnIndex(filePathColumn[0])
            pathToFile = cursor.getString(columnIndex)
            cursor.close()
            val bitmap = BitmapFactory.decodeFile(pathToFile)
            imageView.setImageBitmap(bitmap)

        }
    }

}
