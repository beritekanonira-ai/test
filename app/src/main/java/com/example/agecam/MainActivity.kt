package com.example.agecam

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private val executor = Executors.newSingleThreadExecutor()
    private var useFront = false

    @Volatile private var estimator: AgeEstimator? = null

    private val faceDetector by lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .build()
        )
    }
    private val poseDetector by lazy {
        PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
        )
    }

    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Нужен доступ к камере", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        findViewById<Button>(R.id.flip).setOnClickListener {
            useFront = !useFront
            startCamera()
        }
        executor.execute { estimator = AgeEstimator(applicationContext) }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(executor, ::analyze) }
            val selector =
                if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val raw = proxy.toBitmap()
            val m = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
            val bmp = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            val input = InputImage.fromBitmap(bmp, 0)

            val faces = Tasks.await(faceDetector.process(input))
            val pose = runCatching { Tasks.await(poseDetector.process(input)) }.getOrNull()

            // Person box = bounds of confidently visible body landmarks.
            val person = pose?.allPoseLandmarks
                ?.filter { it.inFrameLikelihood > 0.5f }
                ?.takeIf { it.size >= 5 }
                ?.let { lm ->
                    val pad = 20f
                    RectF(
                        lm.minOf { it.position.x } - pad, lm.minOf { it.position.y } - pad,
                        lm.maxOf { it.position.x } + pad, lm.maxOf { it.position.y } + pad,
                    )
                }

            val labeled = faces.map { face ->
                val r = estimator?.estimate(bmp, face.boundingBox)
                val label = r?.let { "~${it.approxAge} (${it.bucket})" } ?: "лицо"
                OverlayView.Face(RectF(face.boundingBox), label)
            }
            overlay.update(OverlayView.Frame(bmp.width, bmp.height, useFront, person, labeled))
        } catch (e: Exception) {
            // Drop the frame.
        } finally {
            proxy.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
    }
}
