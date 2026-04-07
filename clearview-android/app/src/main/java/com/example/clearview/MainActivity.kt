package com.example.clearview

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import coil.load
import com.example.clearview.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var selectedBitmap: Bitmap? = null
    private var cameraFile: File? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            loadUriToImage(it)
        }
    }

    private val takePhotoLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok && cameraFile != null) {
            val uri = FileProvider.getUriForFile(this, "${applicationContext.packageName}.fileprovider", cameraFile!!)
            loadUriToImage(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUi()
    }

    private fun setupUi() {
        binding.btnSelect.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        binding.btnCamera.setOnClickListener {
            // create temp file
            cameraFile = File.createTempFile("capture_", ".jpg", cacheDir)
            val uri = FileProvider.getUriForFile(this, "${applicationContext.packageName}.fileprovider", cameraFile!!)
            // ask permission for camera via launcher (manifest included)
            takePhotoLauncher.launch(uri)
        }

        // Seekbars: update label and track value
        binding.seekDehaze.setOnSeekBarChangeListener(seekListener { progress ->
            binding.lblDehaze.text = "Dehaze: $progress"
        })
        binding.seekBrightness.setOnSeekBarChangeListener(seekListener { progress ->
            binding.lblBrightness.text = "Brightness: ${progress - 100}"
        })
        binding.seekContrast.setOnSeekBarChangeListener(seekListener { progress ->
            binding.lblContrast.text = "Contrast: ${progress - 100}"
        })
        binding.seekSaturation.setOnSeekBarChangeListener(seekListener { progress ->
            binding.lblSaturation.text = "Saturation: ${progress - 100}"
        })

        binding.btnProcess.setOnClickListener {
            if (selectedBitmap == null) {
                toast("Select or capture an image first")
                return@setOnClickListener
            }
            uploadAndProcess()
        }
    }

    private fun seekListener(onMove: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
            onMove(progress)
        }
        override fun onStartTrackingTouch(seekBar: SeekBar?) {}
        override fun onStopTrackingTouch(seekBar: SeekBar?) {}
    }

    private fun loadUriToImage(uri: Uri) {
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) {
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
            selectedBitmap = bmp
            binding.imgOriginal.load(bmp)
            binding.imgProcessed.setImageDrawable(null)
            binding.txtStatus.text = "Status: ready"
            binding.txtDetections.text = "Detections: none"
        }
    }

    private fun uploadAndProcess() {
        val bmp = selectedBitmap ?: return
        binding.progress.visibility = View.VISIBLE
        binding.txtStatus.text = "Uploading..."

        lifecycleScope.launch {
            try {
                // prepare file
                val file = withContext(Dispatchers.IO) {
                    FileUtils.bitmapToFile(this@MainActivity, bmp, "upload_${System.currentTimeMillis()}.jpg")
                }
                val reqFile = RequestBody.create("image/jpeg".toMediaType(), file)
                val part = MultipartBody.Part.createFormData("file", file.name, reqFile)

                // params map
                val params = hashMapOf<String, RequestBody>()
                // mapping slider values to backend expected names
                val dehaze = (binding.seekDehaze.progress / 100.0).coerceIn(0.1, 2.0)
                params["dehaze_method"] = RequestBody.create("text/plain".toMediaType(), "fast")
                params["dehaze_strength"] = RequestBody.create("text/plain".toMediaType(), dehaze.toString())
                params["histogram_method"] = RequestBody.create("text/plain".toMediaType(), "clahe")
                params["clahe_clip"] = RequestBody.create("text/plain".toMediaType(), "2.0")
                val brightness = (binding.seekBrightness.progress - 100).toString()
                val contrast = (binding.seekContrast.progress - 100).toString()
                val saturation = (binding.seekSaturation.progress - 100).toString()
                params["brightness"] = RequestBody.create("text/plain".toMediaType(), brightness)
                params["contrast"] = RequestBody.create("text/plain".toMediaType(), contrast)
                params["saturation"] = RequestBody.create("text/plain".toMediaType(), saturation)
                params["denoise"] = RequestBody.create("text/plain".toMediaType(), "true")
                params["detect"] = RequestBody.create("text/plain".toMediaType(), "true")

                runOnUiThread {
                    binding.txtStatus.text = "Processing on server..."
                }

                val response = RetrofitClient.api.uploadImage(part, params)

                if (response.isSuccessful && response.body() != null) {
                    val body = response.body()!!
                    if (body.success && body.processed_image != null) {
                        // processed_image is "data:image/jpeg;base64,...."
                        val dataUrl = body.processed_image
                        val base64 = dataUrl.substringAfter("base64,", "")
                        val bytes = android.util.Base64.decode(base64, android.util.Base64.DEFAULT)
                        val bmpProcessed = withContext(Dispatchers.IO) {
                            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        }
                        binding.imgProcessed.load(bmpProcessed)
                        binding.txtStatus.text = "Status: done"
                        binding.txtDetections.text = if (!body.detections.isNullOrEmpty()) {
                            "Detections: " + body.detections.joinToString { "${it.name} ${(it.confidence ?: 0.0 * 100).toInt()}%" }
                        } else {
                            "Detections: none"
                        }
                    } else {
                        binding.txtStatus.text = "Server failed: ${response.errorBody()?.string() ?: "unknown"}"
                    }
                } else {
                    binding.txtStatus.text = "Upload failed: ${response.code()}"
                }

            } catch (ex: Exception) {
                binding.txtStatus.text = "Error: ${ex.message}"
                ex.printStackTrace()
            } finally {
                binding.progress.visibility = View.GONE
            }
        }
    }

    private fun toast(msg: String) {
        android.widget.Toast.makeText(this, msg, android.widget.Toast.LENGTH_SHORT).show()
    }
}