package com.example.clearview

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.*

data class EnhanceResponse(
    val success: Boolean,
    val processed_image: String?, // data URL
    val detections: List<Detection>?
)

data class Detection(
    val name: String?,
    val confidence: Double?,
    val box: List<Int>?
)

interface ApiService {
    // Multipart upload - backend supports 'file' multipart
    @Multipart
    @POST("/api/enhance")
    suspend fun uploadImage(
        @Part file: MultipartBody.Part,
        @PartMap params: Map<String, @JvmSuppressWildcards RequestBody>
    ): Response<EnhanceResponse>
}