package com.example.agecam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.File

/** Levi–Hassner age classifier (Caffe) via OpenCV DNN. */
class AgeEstimator(context: Context) {
    data class Result(val bucket: String, val approxAge: Int, val confidence: Float)

    private val net: Net

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV init failed" }
        val proto = copyAsset(context, "age_deploy.prototxt")
        val model = copyAsset(context, "age_net.caffemodel")
        net = Dnn.readNetFromCaffe(proto.absolutePath, model.absolutePath)
    }

    @Synchronized
    fun estimate(frame: Bitmap, face: Rect): Result? {
        // The network was trained on crops with margin around the face.
        val padX = (face.width() * 0.2f).toInt()
        val padY = (face.height() * 0.2f).toInt()
        val r = Rect(
            (face.left - padX).coerceAtLeast(0),
            (face.top - padY).coerceAtLeast(0),
            (face.right + padX).coerceAtMost(frame.width),
            (face.bottom + padY).coerceAtMost(frame.height),
        )
        if (r.width() < 20 || r.height() < 20) return null

        val crop = Bitmap.createBitmap(frame, r.left, r.top, r.width(), r.height())
        val rgba = Mat()
        val bgr = Mat()
        Utils.bitmapToMat(crop, rgba)
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val blob = Dnn.blobFromImage(bgr, 1.0, Size(227.0, 227.0), MEAN, false, false)
        net.setInput(blob)
        val out = net.forward()

        val p = FloatArray(BUCKETS.size)
        out.get(0, 0, p)
        var best = 0
        var expected = 0f
        var sum = 0f
        for (i in p.indices) {
            if (p[i] > p[best]) best = i
            expected += p[i] * MIDPOINTS[i]
            sum += p[i]
        }
        listOf(rgba, bgr, blob, out).forEach(Mat::release)
        crop.recycle()
        if (sum <= 0f) return null
        return Result(BUCKETS[best], Math.round(expected / sum), p[best] / sum)
    }

    private fun copyAsset(context: Context, name: String): File {
        val f = File(context.cacheDir, name)
        if (!f.exists()) {
            context.assets.open(name).use { i -> f.outputStream().use { o -> i.copyTo(o) } }
        }
        return f
    }

    companion object {
        val BUCKETS = listOf("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+")
        private val MIDPOINTS = floatArrayOf(1f, 5f, 10f, 17.5f, 28.5f, 40.5f, 50.5f, 70f)
        private val MEAN = Scalar(78.4263377603, 87.7689143744, 114.895847746)
    }
}
