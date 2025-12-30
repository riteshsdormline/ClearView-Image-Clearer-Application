package com.example.clearview

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream

object FileUtils {
    fun bitmapToFile(context: Context, bitmap: Bitmap, name: String = "upload.jpg"): File {
        val file = File(context.cacheDir, name)
        file.createNewFile()
        val out = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        out.flush()
        out.close()
        return file
    }
}