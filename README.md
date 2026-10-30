location app>src>main>java>Mainavtivity
package com.wastesorter.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.wastesorter.app.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.min

private const val TAG = "WasteSorter"

/*
 * ESP32 HTTP client.
 *
 * The ESP32 must independently implement a cooldown, duplicate-command
 * protection, safe motor control and an emergency stop.
 */
object Esp32Client {
    private const val BASE_URL = "http://192.168.4.1:5000"
    private const val TIMEOUT_MS = 2500

    @Volatile
    private var wifiNetwork: Network? = null

    fun setWifiNetwork(network: Network?) {
        wifiNetwork = network
    }

    fun clearWifiNetwork(network: Network) {
        if (wifiNetwork == network) {
            wifiNetwork = null
        }
    }

    fun request(path: String): Int {
        require(path.startsWith("/")) {
            "Request path must start with /"
        }

        var connection: HttpURLConnection? = null

        return try {
            val url = URL(BASE_URL + path)
            val network = wifiNetwork

            val urlConnection = if (network != null) {
                network.openConnection(url)
            } else {
                url.openConnection()
            }

            connection = urlConnection as? HttpURLConnection
                ?: error("Could not create HTTP connection")

            connection.requestMethod = "GET"
            connection.connectTimeout = TIMEOUT_MS
            connection.readTimeout = TIMEOUT_MS
            connection.useCaches = false
            connection.instanceFollowRedirects = false

            connection.responseCode
        } catch (e: Exception) {
            Log.e(TAG, "HTTP request failed: $path", e)
            -1
        } finally {
            connection?.disconnect()
        }
    }
}

/*
 * TensorFlow Lite classifier.
 *
 * IMPORTANT:
 * - This implementation requires float32 model input and output.
 * - It assumes the model contains its own input preprocessing.
 * - labels.txt must match the actual output order of the trained model.
 *
 * If your model was trained with different preprocessing, this section
 * must be adapted to that training pipeline.
 */
class WasteClassifier(context: Context) : AutoCloseable {

    data class Result(
        val label: String,
        val confidence: Float,
        val elapsedMs: Long,
        val labels: List<String>,
        val probabilities: FloatArray
    )

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val width: Int
    private val height: Int
    private val classLabels: List<String>
    private val outputCount: Int

    init {
        val modelBytes = context.assets
            .open(MODEL_FILE)
            .use { it.readBytes() }

        require(modelBytes.isNotEmpty()) {
            "waste_model.tflite is empty"
        }

        val modelBuffer = ByteBuffer
            .allocateDirect(modelBytes.size)
            .order(ByteOrder.nativeOrder())

        modelBuffer.put(modelBytes)
        modelBuffer.rewind()

        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().setNumThreads(4)
        )

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()

        require(
            inputShape.size == 4 &&
                inputShape[0] == 1 &&
                inputShape[3] == 3
        ) {
            "Model input must have shape [1,height,width,3]. " +
                "Actual shape: ${inputShape.contentToString()}"
        }

        require(inputTensor.dataType() == DataType.FLOAT32) {
            "The model input must be FLOAT32."
        }

        height = inputShape[1]
        width = inputShape[2]

        require(width > 0 && height > 0) {
            "Invalid model input dimensions."
        }

        inputBuffer = ByteBuffer
            .allocateDirect(inputTensor.numBytes())
            .order(ByteOrder.nativeOrder())

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()

        require(
            outputShape.size == 2 &&
                outputShape[0] == 1
        ) {
            "Model output must have shape [1,classCount]. " +
                "Actual shape: ${outputShape.contentToString()}"
        }

        require(outputTensor.dataType() == DataType.FLOAT32) {
            "The model output must be FLOAT32."
        }

        outputCount = outputShape[1]

        require(outputCount == 3) {
            "Expected three classes; received $outputCount."
        }

        classLabels = context.assets
            .open(LABELS_FILE)
            .bufferedReader()
            .useLines { lines ->
                lines.map {
                    it.trim().lowercase(Locale.ROOT)
                }.filter {
                    it.isNotEmpty()
                }.toList()
            }

        require(classLabels.size == outputCount) {
            "labels.txt must contain exactly $outputCount labels."
        }

        require(classLabels.toSet().size == classLabels.size) {
            "labels.txt contains duplicate labels."
        }

        require(classLabels.toSet() == REQUIRED_LABELS) {
            "labels.txt must contain inorganic, mixed and organic."
        }

        Log.i(
            TAG,
            "Model loaded. Input=${inputShape.contentToString()}, " +
                "labels=$classLabels"
        )
    }

    @Synchronized
    fun classify(bitmap: Bitmap): Result {
        val start = SystemClock.elapsedRealtime()

        val scaled = Bitmap.createScaledBitmap(
            bitmap,
            width,
            height,
            true
        )

        try {
            val pixels = IntArray(width * height)

            scaled.getPixels(
                pixels,
                0,
                width,
                0,
                0,
                width,
                height
            )

            inputBuffer.rewind()

            for (pixel in pixels) {
                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF

                /*
                 * This supplies raw RGB values in [0,255].
                 * Use this only if preprocessing is inside your model.
                 */
                inputBuffer.putFloat(red.toFloat())
                inputBuffer.putFloat(green.toFloat())
                inputBuffer.putFloat(blue.toFloat())
            }

            inputBuffer.rewind()

            val output = Array(1) {
                FloatArray(outputCount)
            }

            interpreter.run(inputBuffer, output)

            val raw = output[0]

            require(raw.all { it.isFinite() }) {
                "The model returned an invalid value."
            }

            val probabilities = normalizeScores(raw)

            var bestIndex = 0

            for (i in 1 until probabilities.size) {
                if (probabilities[i] > probabilities[bestIndex]) {
                    bestIndex = i
                }
            }

            return Result(
                label = classLabels[bestIndex],
                confidence = probabilities[bestIndex],
                elapsedMs = SystemClock.elapsedRealtime() - start,
                labels = classLabels,
                probabilities = probabilities
            )
        } finally {
            if (scaled !== bitmap && !scaled.isRecycled) {
                scaled.recycle()
            }
        }
    }

    private fun normalizeScores(raw: FloatArray): FloatArray {
        val sum = raw.sum()

        if (
            raw.all { it in 0f..1.001f } &&
            sum > 0f &&
            abs(sum - 1f) < 0.15f
        ) {
            return FloatArray(raw.size) { i ->
                (raw[i] / sum).coerceIn(0f, 1f)
            }
        }

        val maximum = raw.maxOrNull() ?: 0f

        val exponents = FloatArray(raw.size) { i ->
            exp((raw[i] - maximum).coerceIn(-80f, 80f))
        }

        val total = exponents.sum().coerceAtLeast(1e-8f)

        return FloatArray(exponents.size) { i ->
            exponents[i] / total
        }
    }

    @Synchronized
    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_FILE = "waste_model.tflite"
        private const val LABELS_FILE = "labels.txt"

        private val REQUIRED_LABELS = setOf(
            "inorganic",
            "mixed",
            "organic"
        )
    }
}

/*
 * Convert CameraX's RGBA_8888 frames to a Bitmap.
 * The camera row stride is respected.
 */
private fun ImageProxy.toBitmap(): Bitmap {
    require(format == android.graphics.PixelFormat.RGBA_8888) {
        "Camera output is not RGBA_8888."
    }

    val plane = planes.firstOrNull()
        ?: error("Camera frame has no pixel plane.")

    val buffer = plane.buffer.duplicate()
    val start = buffer.position()

    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride

    require(pixelStride >= 4) {
        "Invalid camera pixel stride."
    }

    require(rowStride >= width * pixelStride) {
        "Invalid camera row stride."
    }

    val requiredEnd =
        start +
            (height - 1) * rowStride +
            (width - 1) * pixelStride +
            4

    require(requiredEnd <= buffer.limit()) {
        "Camera buffer is shorter than expected."
    }

    val pixels = IntArray(width * height)

    for (y in 0 until height) {
        val rowStart = start + y * rowStride

        for (x in 0 until width) {
            val offset = rowStart + x * pixelStride

            val red = buffer.get(offset).toInt() and 0xFF
            val green = buffer.get(offset + 1).toInt() and 0xFF
            val blue = buffer.get(offset + 2).toInt() and 0xFF
            val alpha = buffer.get(offset + 3).toInt() and 0xFF

            pixels[y * width + x] = Color.argb(
                alpha,
                red,
                green,
                blue
            )
        }
    }

    return Bitmap.createBitmap(
        pixels,
        width,
        height,
        Bitmap.Config.ARGB_8888
    )
}

class MainActivity : AppCompatActivity() {

    private var _binding: ActivityMainBinding? = null
    private val binding: ActivityMainBinding
        get() = checkNotNull(_binding)

    private val mainHandler = Handler(Looper.getMainLooper())

    private val analysisExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val networkExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val ioExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val stateLock = Any()
    private val snapshotLock = Any()

    @Volatile
    private var classifier: WasteClassifier? = null

    @Volatile
    private var analysisRequested = false

    @Volatile
    private var autoAccept = true

    @Volatile
    private var commandInFlight = false

    @Volatile
    private var cycleLocked = false

    @Volatile
    private var destroyed = false

    private var manualChoiceRequired = false
    private var stableLabel: String? = null
    private var stableFrames = 0
    private var analysisDeadline = 0L
    private var lastUiUpdate = 0L
    private var lastSnapshotTime = 0L

    private var camera: Camera? = null
    private var networkCallback: ConnectivityManager.NetworkCallback? = null

    private var manualSnapshot: Bitmap? = null
    private var pendingDecision: Decision? = null

    private var latestResult: WasteClassifier.Result? = null

    private data class Decision(
        val label: String,
        val confidence: Float,
        val source: String
    )

    private val permissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Camera permission is required.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    private val exportLauncher =
        registerForActivityResult(
            ActivityResultContracts.CreateDocument("application/zip")
        ) { uri ->
            if (uri != null) {
                exportFeedback(uri)
            }
        }

    private val unlockRunnable = Runnable {
        synchronized(stateLock) {
            cycleLocked = false
        }

        if (!destroyed && _binding != null) {
            ui {
                sortButton.text = "ANALYZE ITEM"
                sortButton.isEnabled = classifier != null
                classText.text = "READY FOR NEXT ITEM"
                confText.text = "Software cooldown complete"
                metaText.text =
                    "Check that the previous item has cleared the mechanism."
            }

            setStatus("READY", GREEN)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        _binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.hide()

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars()
            )

            val padding =
                (12 * resources.displayMetrics.density).toInt()

            binding.overlay.setPadding(
                bars.left + padding,
                bars.top + padding,
                bars.right + padding,
                bars.bottom + padding
            )

            insets
        }

        binding.autoSwitch.isChecked = true

        binding.autoSwitch.setOnCheckedChangeListener { _, checked ->
            autoAccept = checked

            ui {
                metaText.text = if (checked) {
                    "Stable AI predictions can be accepted automatically."
                } else {
                    "Manual category selection is enabled."
                }
            }
        }

        binding.sortButton.setOnClickListener {
            if (pendingDecision != null) {
                confirmSort()
            } else {
                startAnalysis()
            }
        }

        binding.testButton.setOnClickListener {
            testEsp32()
        }

        binding.exportButton.setOnClickListener {
            exportLauncher.launch("WasteSorterFeedback.zip")
        }

        binding.organicButton.setOnClickListener {
            chooseManualLabel("organic")
        }

        binding.inorganicButton.setOnClickListener {
            chooseManualLabel("inorganic")
        }

        binding.mixedButton.setOnClickListener {
            chooseManualLabel("mixed")
        }

        showManualButtons(false)

        binding.previewView.setOnTouchListener { view, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                try {
                    val point =
                        binding.previewView.meteringPointFactory
                            .createPoint(event.x, event.y)

                    val action = FocusMeteringAction.Builder(point)
                        .setAutoCancelDuration(3, TimeUnit.SECONDS)
                        .build()

                    camera?.cameraControl
                        ?.startFocusAndMetering(action)
                } catch (e: Exception) {
                    Log.w(TAG, "Tap-to-focus failed.", e)
                }

                view.performClick()
            }

            true
        }

        binding.sortButton.isEnabled = false

        loadModel()
        requestEsp32Wifi()

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setStatus("STARTING", AMBER)
        setMessage("Preparing the camera and AI model.")
    }

    private fun ui(block: ActivityMainBinding.() -> Unit) {
        if (destroyed) return

        runOnUiThread {
            val current = _binding ?: return@runOnUiThread
            if (!destroyed) current.block()
        }
    }

    private fun setStatus(text: String, color: String) {
        ui {
            statusChip.text = text
            statusChip.setTextColor(Color.parseColor(color))
        }
    }

    private fun setMessage(text: String) {
        ui {
            metaText.text = text
        }
    }

    private fun loadModel() {
        ioExecutor.execute {
            var loaded: WasteClassifier? = null

            try {
                loaded = WasteClassifier(this)

                if (destroyed) {
                    loaded.close()
                    return@execute
                }

                classifier = loaded

                ui {
                    classText.text = "MODEL READY"
                    confText.text = "Ready to inspect an item"
                    sortButton.text = "ANALYZE ITEM"
                    sortButton.isEnabled = true
                    metaText.text =
                        "Center one item in the inspection square."
                }

                setStatus("MODEL READY", GREEN)
            } catch (e: Exception) {
                try {
                    loaded?.close()
                } catch (_: Exception) {
                }

                Log.e(TAG, "Model loading failed.", e)

                ui {
                    classText.text = "MODEL ERROR"
                    confText.text = "AI unavailable"
                    sortButton.isEnabled = false
                    metaText.text =
                        "Model error: ${e.message ?: e.javaClass.simpleName}"
                }

                setStatus("MODEL ERROR", RED)
            }
        }
    }

    private fun startCamera() {
        if (destroyed || _binding == null) return

        val future = ProcessCameraProvider.getInstance(this)

        future.addListener({
            try {
                if (destroyed || _binding == null) return@addListener

                val provider = future.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(
                            binding.previewView.surfaceProvider
                        )
                    }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

                analysis.setAnalyzer(analysisExecutor) { image ->
                    analyzeFrame(image)
                }

                provider.unbindAll()

                camera = provider.bindToLifecycle(
                    this,
                    androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )

                setStatus("CAMERA READY", GREEN)
            } catch (e: Exception) {
                Log.e(TAG, "Camera startup failed.", e)

                setStatus("CAMERA ERROR", RED)
                setMessage(
                    "Camera error: ${e.message ?: e.javaClass.simpleName}"
                )
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startAnalysis() {
        if (classifier == null) {
            Toast.makeText(
                this,
                "The model is not ready.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        synchronized(stateLock) {
            if (
                analysisRequested ||
                pendingDecision != null ||
                commandInFlight ||
                cycleLocked
            ) {
                Toast.makeText(
                    this,
                    "Finish the current item first.",
                    Toast.LENGTH_SHORT
                ).show()
                return
            }

            analysisRequested = true
        }

        stableLabel = null
        stableFrames = 0
        manualChoiceRequired = false
        latestResult = null
        lastSnapshotTime = 0L

        clearSnapshot()

        analysisDeadline =
            SystemClock.elapsedRealtime() + ANALYSIS_WINDOW_MS

        showManualButtons(false)

        ui {
            sortButton.isEnabled = false
            sortButton.text = "ANALYZING…"
            classText.text = "ANALYZING"
            confText.text = "Up to 10 seconds"
            confBar.progress = 0
            metaText.text =
                "Keep one item centered and hold the camera steady."
        }

        setStatus("ANALYZING ITEM", AMBER)
    }

    private fun analyzeFrame(image: ImageProxy) {
        var bitmap: Bitmap? = null

        try {
            if (destroyed || !analysisRequested) return

            val raw = image.toBitmap()

            bitmap = rotateBitmap(
                raw,
                image.imageInfo.rotationDegrees
            )

            if (bitmap !== raw) {
                raw.recycle()
            }

            val crop = cropInspection(bitmap)

            try {
                val result = classifier?.classify(crop)
                if (result != null && analysisRequested) {
                    handleResult(result, crop)
                }
            } finally {
                crop.recycleQuietly()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Camera analysis error.", e)

            ui {
                metaText.text =
                    "Analysis error: ${e.javaClass.simpleName}"
            }
        } finally {
            bitmap?.recycleQuietly()
            image.close()
        }
    }

    private fun rotateBitmap(
        source: Bitmap,
        degrees: Int
    ): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360

        if (normalized == 0) return source

        val matrix = Matrix().apply {
            postRotate(normalized.toFloat())
        }

        return Bitmap.createBitmap(
            source,
            0,
            0,
            source.width,
            source.height,
            matrix,
            true
        )
    }

    private fun cropInspection(source: Bitmap): Bitmap {
        val side = (
            min(source.width, source.height) * INSPECTION_RATIO
        ).toInt().coerceIn(1, min(source.width, source.height))

        val centerX = source.width / 2
        val centerY = (source.height * INSPECTION_CENTER_Y).toInt()

        val left = (centerX - side / 2)
            .coerceIn(0, source.width - side)

        val top = (centerY - side / 2)
            .coerceIn(0, source.height - side)

        return Bitmap.createBitmap(
            source,
            left,
            top,
            side,
            side
        )
    }

    private fun handleResult(
        result: WasteClassifier.Result,
        image: Bitmap
    ) {
        latestResult = result

        val now = SystemClock.elapsedRealtime()

        if (now - lastSnapshotTime >= SNAPSHOT_INTERVAL_MS) {
            replaceSnapshot(image)
            lastSnapshotTime = now
        }

        val confident = result.confidence >= CONFIDENCE_THRESHOLD

        if (confident && result.label == stableLabel) {
            stableFrames++
        } else {
            stableLabel = if (confident) result.label else null
            stableFrames = if (confident) 1 else 0
        }

        val percent = (
            result.confidence * 100f
        ).toInt().coerceIn(0, 100)

        val scores = result.labels.mapIndexed { index, label ->
            val score = result.probabilities[index] * 100f
            "$label ${score.toInt()}%"
        }.joinToString(" • ")

        if (now - lastUiUpdate >= UI_INTERVAL_MS) {
            lastUiUpdate = now

            ui {
                classText.text =
                    result.label.uppercase(Locale.ROOT)

                classText.setTextColor(
                    categoryColor(result.label)
                )

                confText.text = "CONFIDENCE $percent%"
                confBar.progress = percent

                metaText.text =
                    "$scores • ${result.elapsedMs} ms"
            }
        }

        if (
            confident &&
            stableFrames >= REQUIRED_STABLE_FRAMES &&
            autoAccept
        ) {
            val copy = image.copy(
                Bitmap.Config.ARGB_8888,
                false
            )

            finishDecision(
                result.label,
                result.confidence,
                "AI",
                copy
            )

            return
        }

        if (now >= analysisDeadline) {
            synchronized(stateLock) {
                analysisRequested = false
            }

            manualChoiceRequired = true

            ui {
                sortButton.isEnabled = false
                sortButton.text = "CHOOSE CATEGORY"
                classText.text = "MANUAL DECISION REQUIRED"
                confText.text = "Select the correct category below"
                metaText.text =
                    "Check the item yourself before choosing a label."
            }

            showManualButtons(true)
            setStatus("MANUAL DECISION", AMBER)
        } else if (!confident) {
            showManualButtons(true)
        }
    }

    private fun chooseManualLabel(label: String) {
        if (!analysisRequested && !manualChoiceRequired) {
            return
        }

        val snapshot = copySnapshot()

        if (snapshot == null) {
            Toast.makeText(
                this,
                "No inspection image is available. Analyze again.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        synchronized(stateLock) {
            analysisRequested = false
        }

        manualChoiceRequired = false

        finishDecision(
            label,
            latestResult?.confidence ?: 0f,
            "MANUAL",
            snapshot
        )
    }

    private fun finishDecision(
        label: String,
        confidence: Float,
        source: String,
        image: Bitmap
    ) {
        if (label !in VALID_LABELS) {
            image.recycleQuietly()
            return
        }

        val decision = Decision(
            label = label,
            confidence = confidence.coerceIn(0f, 1f),
            source = source
        )

        val accepted = synchronized(stateLock) {
            if (
                destroyed ||
                pendingDecision != null ||
                commandInFlight ||
                cycleLocked
            ) {
                false
            } else {
                pendingDecision = decision
                true
            }
        }

        if (!accepted) {
            image.recycleQuietly()
            return
        }

        synchronized(stateLock) {
            analysisRequested = false
        }

        manualChoiceRequired = false

        showManualButtons(false)
        clearSnapshot()

        saveTrainingExample(
            image,
            decision.label,
            decision.source,
            decision.confidence
        )

        image.recycleQuietly()

        ui {
            sortButton.text = "CONFIRM SORT"
            sortButton.isEnabled = true

            classText.text =
                "${decision.label.uppercase(Locale.ROOT)} READY"

            classText.setTextColor(
                categoryColor(decision.label)
            )

            confText.text =
                "${decision.source} • " +
                    "${(decision.confidence * 100).toInt()}%"

            metaText.text =
                "Review the category and make sure the mechanism is clear. " +
                    "Tap CONFIRM SORT only when the item is in position."
        }

        setStatus("AWAITING CONFIRMATION", AMBER)
    }

    /*
     * The app does not automatically trigger the pusher.
     * The user must review the prediction and tap CONFIRM SORT.
     */
    private fun confirmSort() {
        val decision: Decision

        synchronized(stateLock) {
            if (
                destroyed ||
                commandInFlight ||
                cycleLocked
            ) {
                return
            }

            decision = pendingDecision ?: return

            pendingDecision = null
            commandInFlight = true
        }

        ui {
            sortButton.isEnabled = false
            sortButton.text = "SENDING…"
            metaText.text =
                "Sending one command. Keep hands clear of the mechanism."
        }

        setStatus("SENDING TO ESP32", AMBER)

        val confidence = String.format(
            Locale.US,
            "%.3f",
            decision.confidence
        )

        val path =
            "/sort?cls=${decision.label}&conf=$confidence"

        try {
            networkExecutor.execute {
                val code = Esp32Client.request(path)
                val serverAccepted = code in 200..299

                synchronized(stateLock) {
                    commandInFlight = false

                    /*
                     * Lock after both success and failure. A timeout does not
                     * prove the ESP32 never received the request.
                     */
                    cycleLocked = true
                }

                mainHandler.removeCallbacks(unlockRunnable)

                if (serverAccepted) {
                    mainHandler.postDelayed(
                        unlockRunnable,
                        CYCLE_LOCK_MS
                    )

                    ui {
                        statusChip.text = "ESP32 ACCEPTED"
                        statusChip.setTextColor(
                            Color.parseColor(GREEN)
                        )

                        sortButton.text = "CYCLE LOCKED"
                        sortButton.isEnabled = false

                        classText.text = "SORTING IN PROGRESS"
                        confText.text = "HTTP $code"

                        metaText.text =
                            "Wait for the full physical cycle. " +
                                "The ESP32 must independently enforce safe timing."
                    }

                    Log.i(TAG, "Sort request accepted: $path")
                } else {
                    /*
                     * Keep locked. Do not retry an uncertain physical action.
                     * Restarting the app is not proof that the mechanism is safe.
                     */
                    ui {
                        statusChip.text = "RESULT UNKNOWN"
                        statusChip.setTextColor(
                            Color.parseColor(RED)
                        )

                        sortButton.text = "CYCLE LOCKED"
                        sortButton.isEnabled = false

                        classText.text = "CHECK MECHANISM"
                        confText.text = "HTTP $code"

                        metaText.text =
                            "No retry was sent. Inspect the mechanism and " +
                                "ESP32 before attempting another cycle."
                    }

                    Log.e(
                        TAG,
                        "Sort request not confirmed: HTTP $code"
                    )
                }
            }
        } catch (e: Exception) {
            synchronized(stateLock) {
                commandInFlight = false
                cycleLocked = true
            }

            ui {
                statusChip.text = "REQUEST ERROR"
                statusChip.setTextColor(
                    Color.parseColor(RED)
                )

                sortButton.text = "CYCLE LOCKED"
                sortButton.isEnabled = false

                classText.text = "CHECK MECHANISM"
                confText.text = e.javaClass.simpleName

                metaText.text =
                    "No retry was sent. Inspect the hardware and ESP32."
            }

            Log.e(TAG, "Could not queue sort request.", e)
        }
    }

    private fun replaceSnapshot(source: Bitmap) {
        val copy = try {
            source.copy(Bitmap.Config.ARGB_8888, false)
        } catch (e: Exception) {
            Log.w(TAG, "Snapshot copy failed.", e)
            null
        } ?: return

        val old = synchronized(snapshotLock) {
            val previous = manualSnapshot
            manualSnapshot = copy
            previous
        }

        old?.recycleQuietly()
    }

    private fun copySnapshot(): Bitmap? {
        return synchronized(snapshotLock) {
            val bitmap = manualSnapshot ?: return@synchronized null

            if (bitmap.isRecycled) {
                return@synchronized null
            }

            try {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)
            } catch (e: Exception) {
                Log.e(TAG, "Snapshot could not be copied.", e)
                null
            }
        }
    }

    private fun clearSnapshot() {
        val old = synchronized(snapshotLock) {
            val previous = manualSnapshot
            manualSnapshot = null
            previous
        }

        old?.recycleQuietly()
    }

    private fun saveTrainingExample(
        bitmap: Bitmap,
        label: String,
        source: String,
        confidence: Float
    ) {
        val copy = try {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } catch (e: Exception) {
            Log.e(TAG, "Training-image copy failed.", e)
            null
        } ?: return

        try {
            ioExecutor.execute {
                try {
                    val root = feedbackRoot()

                    val folder = if (source == "MANUAL") {
                        "manual"
                    } else {
                        "auto_predictions"
                    }

                    val directory = File(
                        root,
                        "$folder/$label"
                    )

                    if (
                        !directory.exists() &&
                        !directory.mkdirs()
                    ) {
                        error("Could not create feedback folder.")
                    }

                    val timestamp = SimpleDateFormat(
                        "yyyyMMdd_HHmmss_SSS",
                        Locale.US
                    ).apply {
                        timeZone = TimeZone.getTimeZone("UTC")
                    }.format(Date())

                    val name = "${timestamp}_${label}.jpg"
                    val file = File(directory, name)

                    FileOutputStream(file).use { output ->
                        if (
                            !copy.compress(
                                Bitmap.CompressFormat.JPEG,
                                92,
                                output
                            )
                        ) {
                            error("JPEG compression failed.")
                        }
                    }

                    val csv = File(root, "metadata.csv")
                    val newFile = !csv.exists()

                    BufferedWriter(
                        OutputStreamWriter(
                            FileOutputStream(csv, true),
                            Charsets.UTF_8
                        )
                    ).use { writer ->
                        if (newFile) {
                            writer.append(
                                "relative_path,label,source," +
                                    "confidence_0_to_1,utc_timestamp\n"
                            )
                        }

                        writer.append(
                            "\"$folder/$label/$name\"," +
                                "$label,$source," +
                                "${String.format(Locale.US, "%.5f", confidence)}," +
                                "$timestamp\n"
                        )
                    }

                    Log.i(TAG, "Saved feedback image: $name")
                } catch (e: Exception) {
                    Log.e(TAG, "Could not save feedback image.", e)

                    ui {
                        metaText.text =
                            "Warning: feedback image could not be saved."
                    }
                } finally {
                    copy.recycleQuietly()
                }
            }
        } catch (e: Exception) {
            copy.recycleQuietly()
            Log.e(TAG, "Could not queue feedback image.", e)
        }
    }

    private fun feedbackRoot(): File {
        val base = getExternalFilesDir(null) ?: filesDir
        return File(base, "WasteSorterFeedback")
    }

    private fun exportFeedback(uri: Uri) {
        setStatus("EXPORTING", AMBER)

        try {
            ioExecutor.execute {
                try {
                    val root = feedbackRoot()

                    val output = contentResolver.openOutputStream(uri)
                        ?: error("Cannot open export destination.")

                    output.use { raw ->
                        ZipOutputStream(raw).use { zip ->
                            if (root.exists()) {
                                root.walkTopDown()
                                    .filter { it.isFile }
                                    .forEach { file ->
                                        val relative = file
                                            .relativeTo(root)
                                            .invariantSeparatorsPath

                                        zip.putNextEntry(
                                            ZipEntry(
                                                "WasteSorterFeedback/$relative"
                                            )
                                        )

                                        file.inputStream().use {
                                            it.copyTo(zip)
                                        }

                                        zip.closeEntry()
                                    }
                            }

                            zip.putNextEntry(
                                ZipEntry("README.txt")
                            )

                            zip.write(
                                (
                                    "WasteSorter feedback export.\n" +
                                        "manual/ contains human-selected labels.\n" +
                                        "auto_predictions/ contains unverified AI labels.\n" +
                                        "Review images before including them in training.\n"
                                    ).toByteArray(Charsets.UTF_8)
                            )

                            zip.closeEntry()
                        }
                    }

                    ui {
                        statusChip.text = "EXPORT COMPLETE"
                        statusChip.setTextColor(
                            Color.parseColor(GREEN)
                        )
                        metaText.text =
                            "Feedback ZIP exported successfully."
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Feedback export failed.", e)

                    ui {
                        metaText.text =
                            "Export failed: ${e.message ?: e.javaClass.simpleName}"
                    }

                    setStatus("EXPORT FAILED", RED)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Could not queue feedback export.", e)
            setStatus("EXPORT FAILED", RED)
        }
    }

    private fun testEsp32() {
        setStatus("TESTING ESP32", AMBER)

        try {
            networkExecutor.execute {
                val code = Esp32Client.request("/ping")

                if (code in 200..299) {
                    setStatus("ESP32 ONLINE", GREEN)

                    setMessage(
                        "ESP32 /ping returned HTTP $code. " +
                            "This does not prove that the motors are safe."
                    )
                } else {
                    setStatus("ESP32 OFFLINE", RED)

                    setMessage(
                        "ESP32 did not respond (HTTP $code). " +
                            "Check its Wi-Fi and server port."
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "ESP32 test could not be queued.", e)
            setStatus("TEST FAILED", RED)
        }
    }

    private fun requestEsp32Wifi() {
        try {
            val manager =
                getSystemService(ConnectivityManager::class.java)

            val request = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                .build()

            val callback =
                object : ConnectivityManager.NetworkCallback() {

                    override fun onAvailable(network: Network) {
                        if (!destroyed) {
                            Esp32Client.setWifiNetwork(network)
                            setStatus("WI-FI AVAILABLE", GREEN)
                        }
                    }

                    override fun onLost(network: Network) {
                        Esp32Client.clearWifiNetwork(network)
                        setStatus("WI-FI DISCONNECTED", RED)
                    }
                }

            networkCallback = callback

            manager.requestNetwork(
                request,
                callback
            )
        } catch (e: Exception) {
            Log.w(TAG, "Wi-Fi request failed.", e)

            setMessage(
                "Join the ESP32 Wi-Fi network in Android settings, " +
                    "then tap TEST ESP32."
            )
        }
    }

    private fun showManualButtons(show: Boolean) {
        ui {
            val visibility = if (show) {
                View.VISIBLE
            } else {
                View.GONE
            }

            manualButtonsRow.visibility = visibility
            organicButton.visibility = visibility
            inorganicButton.visibility = visibility
            mixedButton.visibility = visibility
        }
    }

    private fun categoryColor(label: String): Int {
        return when (label) {
            "organic" -> Color.rgb(0, 255, 156)
            "inorganic" -> Color.rgb(0, 229, 255)
            "mixed" -> Color.rgb(255, 196, 0)
            else -> Color.WHITE
        }
    }

    private fun Bitmap.recycleQuietly() {
        try {
            if (!isRecycled) recycle()
        } catch (_: Exception) {
        }
    }

    override fun onDestroy() {
        destroyed = true

        mainHandler.removeCallbacks(unlockRunnable)

        try {
            val manager =
                getSystemService(ConnectivityManager::class.java)

            networkCallback?.let {
                manager.unregisterNetworkCallback(it)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Network cleanup failed.", e)
        }

        Esp32Client.setWifiNetwork(null)

        synchronized(stateLock) {
            analysisRequested = false
        }

        analysisExecutor.shutdownNow()
        networkExecutor.shutdownNow()
        ioExecutor.shutdown()

        try {
            classifier?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Classifier cleanup failed.", e)
        }

        classifier = null
        clearSnapshot()
        _binding = null

        super.onDestroy()
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.65f
        private const val REQUIRED_STABLE_FRAMES = 3

        private const val ANALYSIS_WINDOW_MS = 10_000L
        private const val SNAPSHOT_INTERVAL_MS = 450L
        private const val UI_INTERVAL_MS = 180L

        /*
         * Software-only cooldown. ESP32 firmware needs its own independent
         * actuator timing and safe interlock.
         */
        private const val CYCLE_LOCK_MS = 13_000L

        private const val INSPECTION_RATIO = 0.54f
        private const val INSPECTION_CENTER_Y = 0.34f

        private const val GREEN = "#00FF9C"
        private const val AMBER = "#FFC400"
        private const val RED = "#FF4D6D"

        private val VALID_LABELS = setOf(
            "organic",
            "inorganic",
            "mixed"
        )
    }
}
