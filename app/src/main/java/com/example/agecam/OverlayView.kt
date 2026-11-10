package com.example.agecam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    data class Face(val box: RectF, val label: String)
    data class Frame(
        val width: Int,
        val height: Int,
        val mirrored: Boolean,
        val person: RectF?,
        val faces: List<Face>,
    )

    @Volatile private var frame: Frame? = null

    private val personPaint = Paint().apply {
        color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val facePaint = Paint().apply {
        color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 6f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE; textSize = 48f; isFakeBoldText = true
    }
    private val bgPaint = Paint().apply { color = Color.argb(160, 0, 0, 0) }

    fun update(f: Frame) {
        frame = f
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val f = frame ?: return
        // PreviewView defaults to FILL_CENTER: scale by the larger ratio and crop.
        val scale = maxOf(width / f.width.toFloat(), height / f.height.toFloat())
        val dx = (width - f.width * scale) / 2f
        val dy = (height - f.height * scale) / 2f

        fun map(r: RectF): RectF {
            val l = r.left * scale + dx
            val rr = r.right * scale + dx
            val t = r.top * scale + dy
            val b = r.bottom * scale + dy
            return if (f.mirrored) RectF(width - rr, t, width - l, b) else RectF(l, t, rr, b)
        }

        f.person?.let { canvas.drawRect(map(it), personPaint) }
        for (face in f.faces) {
            val r = map(face.box)
            canvas.drawRect(r, facePaint)
            val w = textPaint.measureText(face.label)
            val top = (r.top - 60f).coerceAtLeast(0f)
            canvas.drawRect(r.left, top, r.left + w + 20f, top + 60f, bgPaint)
            canvas.drawText(face.label, r.left + 10f, top + 46f, textPaint)
        }
    }
}
