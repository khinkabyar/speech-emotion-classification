// =========================================================
// GLOBAL ELEMENTS
// =========================================================

const homePage = document.getElementById("home");

const audioFileInput = document.getElementById("audioFile");
const dropArea = document.getElementById("dropArea");
const originalAudioPlayer =
  document.getElementById("originalAudioPlayer");

const fileNameEl = document.getElementById("fileName");
const durationEl = document.getElementById("duration");
const sampleRateEl = document.getElementById("sampleRate");

// NEW: Extract Features button
const extractFeaturesBtn =
  document.getElementById("extractFeaturesBtn");

// Classify button
const classifyBtn =
  document.getElementById("classifyBtn");

const emotionResultEl =
  document.getElementById("emotionResult");

const probabilitiesEl =
  document.getElementById("probabilities");

const featureTableEl =
  document.getElementById("featureTable");


// =========================================================
// WAVEFORM ELEMENTS
// =========================================================

const waveformSection =
  document.getElementById("waveformSection");

const originalWaveformCanvas =
  document.getElementById("originalWaveform");

const preEmphasisWaveformCanvas =
  document.getElementById("preEmphasisWaveform");


// =========================================================
// CHART OBJECTS
// =========================================================

let originalWaveformChart = null;
let preEmphasisWaveformChart = null;


// =========================================================
// SETTINGS
// =========================================================

const PRE_EMPHASIS = 0.97;

let selectedFile = null;


// =========================================================
// FEATURE DATA
// =========================================================

// Feature table data
let currentExtractedFeatures = null;

// 360-dimensional vector used for classification
let currentFeatureVector = null;

// Active feature tab
let activeFeatureTab = "MFCC";


// =========================================================
// 1. START SYSTEM
// =========================================================

function startSystem() {

    if (homePage) {

        homePage.style.transition =
            "opacity 0.4s ease";

        homePage.style.opacity = "0";

        setTimeout(() => {

            homePage.style.display = "none";

        }, 400);
    }
}


// =========================================================
// 2. SIDEBAR NAVIGATION
// =========================================================

function navigateTo(targetId) {

    // Go back to Home
    if (targetId === "home") {

        if (homePage) {

            homePage.style.display = "flex";
            homePage.style.opacity = "1";
        }

        return;
    }


    // Find target section
    const targetElement =
        document.getElementById(targetId);

    if (targetElement) {

        targetElement.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }


    // Remove active state
    document
        .querySelectorAll(".sidebar .menu-item")
        .forEach((item) => {

            item.classList.remove("active");

        });


    // Add active state
    if (targetId === "upload-section") {

        document
            .getElementById("menuUpload")
            ?.classList.add("active");

    } else if (targetId === "features-section") {

        document
            .getElementById("menuFeatures")
            ?.classList.add("active");

    } else if (targetId === "settings-section") {

        document
            .getElementById("menuSettings")
            ?.classList.add("active");

    } else if (targetId === "about-section") {

        document
            .getElementById("menuAbout")
            ?.classList.add("active");
    }
}


// =========================================================
// 3. FILE SELECTION
// =========================================================

if (audioFileInput) {

    audioFileInput.addEventListener(
        "change",
        (e) => {

            const file =
                e.target.files[0];

            if (file) {

                handleSelectedFile(file);
            }
        }
    );
}


// =========================================================
// 4. DRAG & DROP
// =========================================================

if (dropArea) {

    // Prevent browser default behavior
    [
        "dragenter",
        "dragover",
        "dragleave",
        "drop"
    ].forEach((eventName) => {

        dropArea.addEventListener(
            eventName,
            (e) => {

                e.preventDefault();
                e.stopPropagation();

            },
            false
        );
    });


    // Highlight drop area
    [
        "dragenter",
        "dragover"
    ].forEach((eventName) => {

        dropArea.addEventListener(
            eventName,
            () => {

                dropArea.classList.add(
                    "highlight"
                );

            }
        );
    });


    // Remove highlight
    [
        "dragleave",
        "drop"
    ].forEach((eventName) => {

        dropArea.addEventListener(
            eventName,
            () => {

                dropArea.classList.remove(
                    "highlight"
                );

            }
        );
    });


    // Handle dropped file
    dropArea.addEventListener(
        "drop",
        (e) => {

            const file =
                e.dataTransfer.files[0];

            if (file) {

                handleSelectedFile(file);
            }
        }
    );
}


// =========================================================
// 5. HANDLE SELECTED FILE
// =========================================================

function handleSelectedFile(file) {

    selectedFile = file;


    // -------------------------------------------------------
    // File Name
    // -------------------------------------------------------

    if (fileNameEl) {

        fileNameEl.textContent =
            file.name;
    }


    // -------------------------------------------------------
    // Set Audio Player
    // -------------------------------------------------------

    if (originalAudioPlayer) {

        const audioURL =
            URL.createObjectURL(file);

        originalAudioPlayer.src =
            audioURL;
    }


    // -------------------------------------------------------
    // Reset Previous Prediction
    // -------------------------------------------------------

    if (emotionResultEl) {

        emotionResultEl.textContent =
            "---";
    }

    if (probabilitiesEl) {

        probabilitiesEl.innerHTML = "";
    }


    // -------------------------------------------------------
    // Reset Previous Features
    // -------------------------------------------------------

    currentExtractedFeatures = null;
    currentFeatureVector = null;

    if (featureTableEl) {

        featureTableEl.innerHTML = "";
    }


    // -------------------------------------------------------
    // Reset Feature Status
    // -------------------------------------------------------

    const featureStatusText =
        document.getElementById(
            "featureStatusText"
        );

    if (featureStatusText) {

        featureStatusText.textContent =
            "Please extract features from the audio file.";
    }


    // -------------------------------------------------------
    // Button State Before Audio Loading
    // -------------------------------------------------------

    if (extractFeaturesBtn) {

        extractFeaturesBtn.disabled = true;

        extractFeaturesBtn.textContent =
            "📊 Extract Features";
    }

    if (classifyBtn) {

        classifyBtn.disabled = true;

        classifyBtn.textContent =
            "🎯 Classify Emotion";
    }


    // -------------------------------------------------------
    // Read Audio
    // -------------------------------------------------------

    const AudioContext =
        window.AudioContext ||
        window.webkitAudioContext;

    if (!AudioContext) {

        alert(
            "Web Audio API is not supported by this browser."
        );

        return;
    }


    const audioContext =
        new AudioContext();

    const reader =
        new FileReader();


    reader.onload = function (e) {

        audioContext
            .decodeAudioData(e.target.result)

            .then((buffer) => {

                // -------------------------------------------
                // Audio Information
                // -------------------------------------------

                if (durationEl) {

                    durationEl.textContent =
                        `${buffer.duration.toFixed(2)} seconds`;
                }

                if (sampleRateEl) {

                    sampleRateEl.textContent =
                        `${buffer.sampleRate} Hz`;
                }


                // -------------------------------------------
                // Create Waveforms
                // -------------------------------------------

                createWaveforms(buffer);


                // -------------------------------------------
                // IMPORTANT BUTTON FLOW
                //
                // Audio loaded successfully:
                // Extract = ENABLED
                // Classify = DISABLED
                // -------------------------------------------

                if (extractFeaturesBtn) {

                    extractFeaturesBtn.disabled =
                        false;
                }

                if (classifyBtn) {

                    classifyBtn.disabled =
                        true;
                }


                // -------------------------------------------
                // Close AudioContext
                // -------------------------------------------

                if (
                    audioContext.state !==
                    "closed"
                ) {

                    audioContext.close();
                }

            })

            .catch((error) => {

                console.error(
                    "Audio decoding error:",
                    error
                );


                if (durationEl) {

                    durationEl.textContent =
                        "Unknown";
                }

                if (sampleRateEl) {

                    sampleRateEl.textContent =
                        "Unknown";
                }


                if (extractFeaturesBtn) {

                    extractFeaturesBtn.disabled =
                        true;
                }

                if (classifyBtn) {

                    classifyBtn.disabled =
                        true;
                }


                if (
                    audioContext.state !==
                    "closed"
                ) {

                    audioContext.close();
                }


                alert(
                    "Unable to read the selected audio file."
                );
            });
    };


    reader.onerror = function () {

        alert(
            "Unable to read the audio file."
        );

        if (extractFeaturesBtn) {

            extractFeaturesBtn.disabled = true;
        }

        if (classifyBtn) {

            classifyBtn.disabled = true;
        }
    };


    reader.readAsArrayBuffer(file);
}


// =========================================================
// 6. CREATE ORIGINAL + PRE-EMPHASIZED WAVEFORMS
// =========================================================

function createWaveforms(buffer) {

    if (
        !originalWaveformCanvas ||
        !preEmphasisWaveformCanvas
    ) {

        console.error(
            "Waveform canvas not found."
        );

        return;
    }


    // -------------------------------------------------------
    // Get Original Audio Samples
    // -------------------------------------------------------

    const channelData =
        buffer.getChannelData(0);

    const sampleRate =
        buffer.sampleRate;

    const totalSamples =
        channelData.length;


    // -------------------------------------------------------
    // APPLY PRE-EMPHASIS
    //
    // y[n] = x[n] - alpha*x[n-1]
    //
    // alpha = 0.97
    // -------------------------------------------------------

    const preEmphasizedData =
        new Float32Array(totalSamples);


    // First sample
    if (totalSamples > 0) {

        preEmphasizedData[0] =
            channelData[0];
    }


    // Remaining samples
    for (
        let i = 1;
        i < totalSamples;
        i++
    ) {

        preEmphasizedData[i] =
            channelData[i] -
            PRE_EMPHASIS *
            channelData[i - 1];
    }


    // -------------------------------------------------------
    // DOWN-SAMPLE ONLY FOR DISPLAY
    //
    // Pre-emphasis is calculated on ALL samples.
    // Down-sampling is only for chart display.
    // -------------------------------------------------------

    const maxPoints = 2000;

    const step =
        Math.max(
            1,
            Math.ceil(
                totalSamples /
                maxPoints
            )
        );


    const originalData = [];
    const preEmphasisData = [];


    for (
        let i = 0;
        i < totalSamples;
        i += step
    ) {

        const time =
            i / sampleRate;


        originalData.push({

            x: time,
            y: channelData[i]

        });


        preEmphasisData.push({

            x: time,
            y: preEmphasizedData[i]

        });
    }


    // -------------------------------------------------------
    // SHOW WAVEFORM SECTION
    // -------------------------------------------------------

    if (waveformSection) {

        waveformSection.style.display =
            "block";
    }


    // -------------------------------------------------------
    // DESTROY OLD CHARTS
    // -------------------------------------------------------

    if (originalWaveformChart) {

        originalWaveformChart.destroy();

        originalWaveformChart = null;
    }


    if (preEmphasisWaveformChart) {

        preEmphasisWaveformChart.destroy();

        preEmphasisWaveformChart = null;
    }


    // -------------------------------------------------------
    // COMMON CHART OPTIONS
    // -------------------------------------------------------

    const commonOptions = {

        responsive: true,

        maintainAspectRatio: false,

        animation: false,

        interaction: {

            intersect: false,

            mode: "nearest"
        },


        scales: {

            x: {

                type: "linear",

                min: 0,

                max: buffer.duration,

                title: {

                    display: true,

                    text:
                        "Time (seconds)"
                },


                ticks: {

                    stepSize: 1,

                    callback:
                        function (value) {

                            return Number(
                                value
                            ).toFixed(0);
                        },

                    maxTicksLimit: 7
                }
            },


            y: {

                title: {

                    display: true,

                    text:
                        "Amplitude"
                }
            }
        },


        plugins: {

            legend: {

                display: false
            },


            tooltip: {

                callbacks: {

                    title:
                        function (
                            tooltipItems
                        ) {

                            const x =
                                tooltipItems[0]
                                    .parsed.x;

                            return (
                                "Time: " +
                                x.toFixed(3) +
                                " seconds"
                            );
                        },


                    label:
                        function (
                            context
                        ) {

                            return (
                                "Amplitude: " +
                                context.parsed.y
                                    .toFixed(4)
                            );
                        }
                }
            }
        }
    };


    // =======================================================
    // ORIGINAL WAVEFORM
    // =======================================================

    originalWaveformChart =
        new Chart(
            originalWaveformCanvas,
            {

                type: "line",

                data: {

                    datasets: [

                        {

                            label:
                                "Original Waveform",

                            data:
                                originalData,

                            borderWidth: 1,

                            pointRadius: 0,

                            tension: 0,

                            fill: false
                        }
                    ]
                },

                options:
                    commonOptions
            }
        );


    // =======================================================
    // PRE-EMPHASIZED WAVEFORM
    // =======================================================

    preEmphasisWaveformChart =
        new Chart(
            preEmphasisWaveformCanvas,
            {

                type: "line",

                data: {

                    datasets: [

                        {

                            label:
                                "Pre-emphasized Waveform",

                            data:
                                preEmphasisData,

                            borderWidth: 1,

                            pointRadius: 0,

                            tension: 0,

                            fill: false
                        }
                    ]
                },

                options:
                    commonOptions
            }
        );
}


// =========================================================
// 7. EXTRACT FEATURES
// =========================================================

if (extractFeaturesBtn) {

    extractFeaturesBtn.addEventListener(
        "click",
        async () => {

            // -----------------------------------------------
            // Check file
            // -----------------------------------------------

            if (!selectedFile) {

                alert(
                    "Please select an audio file first."
                );

                return;
            }


            // -----------------------------------------------
            // Button state
            // -----------------------------------------------

            extractFeaturesBtn.disabled =
                true;

            extractFeaturesBtn.textContent =
                "⏳ Extracting...";


            if (classifyBtn) {

                classifyBtn.disabled =
                    true;
            }


            // -----------------------------------------------
            // Feature status
            // -----------------------------------------------

            const featureStatusText =
                document.getElementById(
                    "featureStatusText"
                );

            if (featureStatusText) {

                featureStatusText.textContent =
                    "Extracting MFCC, Chroma, and Mel-Spectrogram features...";
            }


            // -----------------------------------------------
            // Clear old features
            // -----------------------------------------------

            currentExtractedFeatures =
                null;

            currentFeatureVector =
                null;


            if (featureTableEl) {

                featureTableEl.innerHTML =
                    "";
            }


            // -----------------------------------------------
            // Prepare FormData
            // -----------------------------------------------

            const formData =
                new FormData();

            formData.append(
                "audio",
                selectedFile
            );


            try {

                // -------------------------------------------
                // Send audio to Flask
                // -------------------------------------------

                const response =
                    await fetch(
                        "/extract_features",
                        {

                            method: "POST",

                            body: formData
                        }
                    );


                // -------------------------------------------
                // Read response safely
                // -------------------------------------------

                const contentType =
                    response.headers.get(
                        "content-type"
                    ) || "";


                let data;


                if (
                    contentType.includes(
                        "application/json"
                    )
                ) {

                    data =
                        await response.json();

                } else {

                    const text =
                        await response.text();

                    throw new Error(
                        `Server returned ${response.status}: ${text.substring(0, 200)}`
                    );
                }


                // -------------------------------------------
                // Successful extraction
                // -------------------------------------------

                if (
                    response.ok &&
                    data.success
                ) {

                    currentExtractedFeatures =
                        data.extracted_features;

                    currentFeatureVector =
                        data.feature_vector;


                    // Check feature vector
                    if (
                        !Array.isArray(
                            currentFeatureVector
                        )
                    ) {

                        throw new Error(
                            "Feature vector was not returned by the server."
                        );
                    }


                    if (
                        currentFeatureVector.length !==
                        360
                    ) {

                        throw new Error(
                            `Expected 360 features, but received ${currentFeatureVector.length}.`
                        );
                    }


                    // ---------------------------------------
                    // Render feature table
                    // ---------------------------------------

                    renderFeatureTableByTab(
                        activeFeatureTab
                    );


                    // ---------------------------------------
                    // Update status
                    // ---------------------------------------

                    if (featureStatusText) {

                        featureStatusText.textContent =
                            "360 acoustic features extracted successfully.";
                    }


                    // ---------------------------------------
                    // Enable classification
                    // ---------------------------------------

                    if (classifyBtn) {

                        classifyBtn.disabled =
                            false;
                    }

                } else {

                    throw new Error(
                        data.error ||
                        "Failed to extract acoustic features."
                    );
                }


            } catch (error) {

                console.error(
                    "Feature extraction error:",
                    error
                );


                currentExtractedFeatures =
                    null;

                currentFeatureVector =
                    null;


                if (featureTableEl) {

                    featureTableEl.innerHTML =
                        "";
                }


                if (featureStatusText) {

                    featureStatusText.textContent =
                        "Feature extraction failed.";
                }


                alert(
                    "Feature Extraction Error:\n\n" +
                    error.message
                );


                if (classifyBtn) {

                    classifyBtn.disabled =
                        true;
                }

            } finally {

                // -------------------------------------------
                // Restore Extract button
                // -------------------------------------------

                extractFeaturesBtn.disabled =
                    false;

                extractFeaturesBtn.textContent =
                    "📊 Extract Features";
            }
        }
    );
}


// =========================================================
// 8. CLASSIFY EMOTION
// =========================================================

if (classifyBtn) {

    classifyBtn.addEventListener(
        "click",
        async () => {

            // -----------------------------------------------
            // IMPORTANT:
            // Classification requires extracted features
            // -----------------------------------------------

            if (!currentFeatureVector) {

                alert(
                    "Please click 'Extract Features' first."
                );

                return;
            }


            if (
                !Array.isArray(
                    currentFeatureVector
                )
            ) {

                alert(
                    "Invalid feature vector."
                );

                return;
            }


            if (
                currentFeatureVector.length !==
                360
            ) {

                alert(
                    "The feature vector must contain 360 features."
                );

                return;
            }


            // -----------------------------------------------
            // Disable button
            // -----------------------------------------------

            classifyBtn.disabled =
                true;

            classifyBtn.textContent =
                "⏳ Classifying...";


            // -----------------------------------------------
            // Show processing state
            // -----------------------------------------------

            if (emotionResultEl) {

                emotionResultEl.textContent =
                    "Analyzing...";
            }


            if (probabilitiesEl) {

                probabilitiesEl.innerHTML =
                    "<p>Processing classification...</p>";
            }


            try {

                // -------------------------------------------
                // Send FEATURE VECTOR to Flask
                //
                // Do NOT send audio again.
                // -------------------------------------------

                const response =
                    await fetch(
                        "/predict",
                        {

                            method: "POST",

                            headers: {

                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({

                                    feature_vector:
                                        currentFeatureVector
                                })
                        }
                    );


                // -------------------------------------------
                // Read response safely
                // -------------------------------------------

                const contentType =
                    response.headers.get(
                        "content-type"
                    ) || "";


                let data;


                if (
                    contentType.includes(
                        "application/json"
                    )
                ) {

                    data =
                        await response.json();

                } else {

                    const text =
                        await response.text();

                    throw new Error(
                        `Server returned ${response.status}: ${text.substring(0, 200)}`
                    );
                }


                // -------------------------------------------
                // Successful prediction
                // -------------------------------------------

                if (
                    response.ok &&
                    data.success
                ) {

                    // ---------------------------------------
                    // Prediction
                    // ---------------------------------------

                    displayPredictionResult(
                        data.emotion,
                        data.confidence
                    );


                    // ---------------------------------------
                    // Probabilities
                    // ---------------------------------------

                    if (
                        data.probabilities
                    ) {

                        displayProbabilities(
                            data.probabilities
                        );
                    }

                } else {

                    throw new Error(
                        data.error ||
                        "Failed to classify the audio."
                    );
                }


            } catch (error) {

                console.error(
                    "Classification error:",
                    error
                );


                alert(
                    "Classification Error:\n\n" +
                    error.message
                );


                if (emotionResultEl) {

                    emotionResultEl.textContent =
                        "Error";
                }

            } finally {

                // -------------------------------------------
                // Restore Classify button
                // -------------------------------------------

                classifyBtn.disabled =
                    false;

                classifyBtn.textContent =
                    "🎯 Classify Emotion";
            }
        }
    );
}


// =========================================================
// 9. DISPLAY PREDICTION RESULT
// =========================================================

function displayPredictionResult(
    emotion,
    confidence
) {

    if (!emotionResultEl) {

        return;
    }


    emotionResultEl.innerHTML = `

        <span
            style="
                color: #2b5cff;
                text-transform: capitalize;
            "
        >
            ${emotion}
        </span>

        <div
            style="
                font-size: 0.9rem;
                font-weight: normal;
                color: #666;
                margin-top: 5px;
            "
        >

            Confidence:

            <strong>
                ${Number(confidence).toFixed(2)}%
            </strong>

        </div>

    `;
}


// =========================================================
// 10. DISPLAY PROBABILITIES
// =========================================================

function displayProbabilities(
    probabilities
) {

    if (!probabilitiesEl) {

        return;
    }


    probabilitiesEl.innerHTML =
        "";


    // Sort highest to lowest
    const sorted =
        Object.entries(
            probabilities
        ).sort(
            (a, b) => b[1] - a[1]
        );


    sorted.forEach(
        ([emotion, prob]) => {

            const item =
                document.createElement(
                    "div"
                );


            item.style.marginBottom =
                "10px";


            item.innerHTML = `

                <div
                    style="
                        display: flex;
                        justify-content: space-between;
                        font-size: 0.85rem;
                        margin-bottom: 3px;
                    "
                >

                    <span
                        style="
                            text-transform: capitalize;
                        "
                    >
                        ${emotion}
                    </span>

                    <span>
                        ${Number(prob).toFixed(2)}%
                    </span>

                </div>


                <div
                    style="
                        background-color: #e2e8f0;
                        border-radius: 6px;
                        height: 8px;
                        overflow: hidden;
                    "
                >

                    <div
                        style="
                            width: ${Number(prob)}%;
                            background-color: #2b5cff;
                            height: 100%;
                            transition: width 0.5s ease;
                        "
                    >
                    </div>

                </div>

            `;


            probabilitiesEl.appendChild(
                item
            );
        }
    );
}


// =========================================================
// 11. FEATURE TAB SWITCHING
// =========================================================

function switchFeatureTab(tabName) {

    activeFeatureTab =
        tabName;


    // Remove active state
    document
        .querySelectorAll(".gui-tab-btn")
        .forEach((btn) => {

            btn.classList.remove(
                "active"
            );

        });


    // Add active state
    const activeBtn =
        document.getElementById(
            `btn${tabName}`
        );


    if (activeBtn) {

        activeBtn.classList.add(
            "active"
        );
    }


    // Render selected table
    renderFeatureTableByTab(
        tabName
    );
}


// =========================================================
// 12. RENDER FEATURE TABLE BY TAB
// =========================================================

function renderFeatureTableByTab(
    tabName
) {

    const table =
        document.getElementById(
            "featureTable"
        );

    const featureStatusText =
        document.getElementById(
            "featureStatusText"
        );


    if (
        !table ||
        !featureStatusText
    ) {

        return;
    }


    // Clear table
    table.innerHTML =
        "";


    // -------------------------------------------------------
    // NO FEATURES YET
    // -------------------------------------------------------

    if (!currentExtractedFeatures) {

        featureStatusText.textContent =
            "Please extract features first.";

        return;
    }


    // -------------------------------------------------------
    // SELECT FEATURE PREFIX
    // -------------------------------------------------------

    let prefix = "";
    let expectedCount = 0;
    let displayName = "";


    if (tabName === "MFCC") {

        prefix = "MFCC_";

        expectedCount = 80;

        displayName = "MFCC";

    } else if (
        tabName === "Chroma"
    ) {

        prefix = "Chroma_";

        expectedCount = 24;

        displayName = "CHROMA";

    } else if (
        tabName === "Mel"
    ) {

        prefix = "Mel_";

        expectedCount = 256;

        displayName =
            "MEL SPECTROGRAM";
    }


    // -------------------------------------------------------
    // FILTER FEATURES
    // -------------------------------------------------------

    const filteredKeys =
        Object.keys(
            currentExtractedFeatures
        )
        .filter(
            (key) =>
                key.startsWith(prefix)
        )
        .sort(
            (a, b) => {

                // Get feature number
                const numberA =
                    parseInt(
                        a.match(
                            /_(\d+)_/
                        )?.[1] || "0"
                    );

                const numberB =
                    parseInt(
                        b.match(
                            /_(\d+)_/
                        )?.[1] || "0"
                    );


                // Sort by feature number
                if (
                    numberA !==
                    numberB
                ) {

                    return (
                        numberA -
                        numberB
                    );
                }


                // Mean first
                // Std second
                const typeA =
                    a.endsWith(
                        "_mean"
                    )
                        ? 0
                        : 1;

                const typeB =
                    b.endsWith(
                        "_mean"
                    )
                        ? 0
                        : 1;


                return (
                    typeA -
                    typeB
                );
            }
        );


    // -------------------------------------------------------
    // STATUS TEXT
    // -------------------------------------------------------

    featureStatusText.textContent =
        `Showing ${filteredKeys.length} ${displayName} features`;


    // -------------------------------------------------------
    // OPTIONAL COUNT CHECK
    // -------------------------------------------------------

    if (
        filteredKeys.length !==
        expectedCount
    ) {

        console.warn(
            `${displayName}: expected ${expectedCount}, received ${filteredKeys.length}`
        );
    }


    // -------------------------------------------------------
    // DISPLAY FEATURES
    // -------------------------------------------------------

    filteredKeys.forEach(
        (key) => {

            const rawValue =
                currentExtractedFeatures[
                    key
                ];


            const value =
                rawValue !== undefined &&
                rawValue !== null &&
                !Number.isNaN(
                    Number(rawValue)
                )
                    ? Number(
                        rawValue
                    ).toFixed(4)
                    : "-";


            const type =
                key.endsWith("_std")
                    ? "Std"
                    : "Mean";


            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML = `

                <td>
                    <strong>
                        ${key}
                    </strong>
                </td>

                <td>
                    ${type}
                </td>

                <td
                    style="
                        text-align: right;
                        font-family: monospace;
                        color: #1f6bba;
                        font-weight: bold;
                    "
                >
                    ${value}
                </td>

            `;


            table.appendChild(
                row
            );
        }
    );
}


// =========================================================
// 13. CLEAR ALL
// =========================================================

function clearAll() {

    // -------------------------------------------------------
    // Clear Selected File
    // -------------------------------------------------------

    selectedFile = null;


    // -------------------------------------------------------
    // Clear Feature Data
    // -------------------------------------------------------

    currentExtractedFeatures =
        null;

    currentFeatureVector =
        null;

    activeFeatureTab =
        "MFCC";


    // -------------------------------------------------------
    // Clear File Input
    // -------------------------------------------------------

    if (audioFileInput) {

        audioFileInput.value =
            "";
    }


    // -------------------------------------------------------
    // Clear Audio Player
    // -------------------------------------------------------

    if (originalAudioPlayer) {

        originalAudioPlayer.pause();

        originalAudioPlayer.src =
            "";

        originalAudioPlayer.load();
    }


    // -------------------------------------------------------
    // Destroy Original Waveform
    // -------------------------------------------------------

    if (originalWaveformChart) {

        originalWaveformChart.destroy();

        originalWaveformChart =
            null;
    }


    // -------------------------------------------------------
    // Destroy Pre-emphasis Waveform
    // -------------------------------------------------------

    if (preEmphasisWaveformChart) {

        preEmphasisWaveformChart.destroy();

        preEmphasisWaveformChart =
            null;
    }


    // -------------------------------------------------------
    // Hide Waveform Section
    // -------------------------------------------------------

    if (waveformSection) {

        waveformSection.style.display =
            "none";
    }


    // -------------------------------------------------------
    // Clear Audio Information
    // -------------------------------------------------------

    if (fileNameEl) {

        fileNameEl.textContent =
            "No file selected";
    }


    if (durationEl) {

        durationEl.textContent =
            "-";
    }


    if (sampleRateEl) {

        sampleRateEl.textContent =
            "-";
    }


    // -------------------------------------------------------
    // Clear Prediction
    // -------------------------------------------------------

    if (emotionResultEl) {

        emotionResultEl.textContent =
            "---";
    }


    if (probabilitiesEl) {

        probabilitiesEl.innerHTML =
            "";
    }


    // -------------------------------------------------------
    // Clear Feature Table
    // -------------------------------------------------------

    if (featureTableEl) {

        featureTableEl.innerHTML =
            "";
    }


    // -------------------------------------------------------
    // Reset Feature Status
    // -------------------------------------------------------

    const featureStatusText =
        document.getElementById(
            "featureStatusText"
        );


    if (featureStatusText) {

        featureStatusText.textContent =
            "Please upload an audio file to begin.";
    }


    // -------------------------------------------------------
    // Reset Active Tab
    // -------------------------------------------------------

    document
        .querySelectorAll(".gui-tab-btn")
        .forEach((btn) => {

            btn.classList.remove(
                "active"
            );

        });


    const defaultTab =
        document.getElementById(
            "btnMFCC"
        );


    if (defaultTab) {

        defaultTab.classList.add(
            "active"
        );
    }


    // -------------------------------------------------------
    // Disable Extract Button
    // -------------------------------------------------------

    if (extractFeaturesBtn) {

        extractFeaturesBtn.disabled =
            true;

        extractFeaturesBtn.textContent =
            "📊 Extract Features";
    }


    // -------------------------------------------------------
    // Disable Classify Button
    // -------------------------------------------------------

    if (classifyBtn) {

        classifyBtn.disabled =
            true;

        classifyBtn.textContent =
            "🎯 Classify Emotion";
    }
}


// =========================================================
// 14. PAGE LOAD
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        currentExtractedFeatures =
            null;

        currentFeatureVector =
            null;

        activeFeatureTab =
            "MFCC";


        // Set MFCC tab active
        const defaultTab =
            document.getElementById(
                "btnMFCC"
            );


        if (defaultTab) {

            defaultTab.classList.add(
                "active"
            );
        }


        // Initial button state
        if (extractFeaturesBtn) {

            extractFeaturesBtn.disabled =
                true;
        }


        if (classifyBtn) {

            classifyBtn.disabled =
                true;
        }
    }
);

// // =========================================================
// // GLOBAL ELEMENTS
// // =========================================================

// const homePage = document.getElementById("home");

// const audioFileInput = document.getElementById("audioFile");
// const dropArea = document.getElementById("dropArea");
// const originalAudioPlayer = document.getElementById("originalAudioPlayer");

// const fileNameEl = document.getElementById("fileName");
// const durationEl = document.getElementById("duration");
// const sampleRateEl = document.getElementById("sampleRate");
// const extractFeaturesBtn = document.getElementById("extractFeaturesBtn");
// const classifyBtn = document.getElementById("classifyBtn");

// const emotionResultEl = document.getElementById("emotionResult");
// const probabilitiesEl = document.getElementById("probabilities");
// const featureTableEl = document.getElementById("featureTable");

// // =========================================================
// // WAVEFORM ELEMENTS
// // =========================================================

// const waveformSection = document.getElementById("waveformSection");

// const originalWaveformCanvas = document.getElementById("originalWaveform");

// const preEmphasisWaveformCanvas = document.getElementById(
//   "preEmphasisWaveform",
// );

// // =========================================================
// // CHART OBJECTS
// // =========================================================

// let originalWaveformChart = null;
// let preEmphasisWaveformChart = null;

// // =========================================================
// // SETTINGS
// // =========================================================

// const PRE_EMPHASIS = 0.97;

// let selectedFile = null;

// // =========================================================
// // FEATURE DATA
// // =========================================================

// // Stores extracted feature data returned from Flask
// let currentExtractedFeatures = null;
// let currentFeatureVector = null;

// // Default active tab
// let activeFeatureTab = "MFCC";

// // =========================================================
// // 1. START SYSTEM
// // =========================================================

// function startSystem() {
//   if (homePage) {
//     homePage.style.transition = "opacity 0.4s ease";
//     homePage.style.opacity = "0";

//     setTimeout(() => {
//       homePage.style.display = "none";
//     }, 400);
//   }
// }

// // =========================================================
// // 2. SIDEBAR NAVIGATION
// // =========================================================

// function navigateTo(targetId) {
//   // Go back to Home
//   if (targetId === "home") {
//     if (homePage) {
//       homePage.style.display = "flex";
//       homePage.style.opacity = "1";
//     }

//     return;
//   }

//   // Find target section
//   const targetElement = document.getElementById(targetId);

//   if (targetElement) {
//     targetElement.scrollIntoView({
//       behavior: "smooth",
//       block: "start",
//     });
//   }

//   // Remove active state
//   document.querySelectorAll(".sidebar .menu-item").forEach((item) => {
//     item.classList.remove("active");
//   });

//   // Add active state
//   if (targetId === "upload-section") {
//     document.getElementById("menuUpload")?.classList.add("active");
//   } else if (targetId === "features-section") {
//     document.getElementById("menuFeatures")?.classList.add("active");
//   } else if (targetId === "settings-section") {
//     document.getElementById("menuSettings")?.classList.add("active");
//   } else if (targetId === "about-section") {
//     document.getElementById("menuAbout")?.classList.add("active");
//   }
// }

// // =========================================================
// // 3. FILE SELECTION
// // =========================================================

// if (audioFileInput) {
//   audioFileInput.addEventListener("change", (e) => {
//     const file = e.target.files[0];

//     if (file) {
//       handleSelectedFile(file);
//     }
//   });
// }

// // =========================================================
// // 4. DRAG & DROP
// // =========================================================

// if (dropArea) {
//   // Prevent browser default behavior
//   ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
//     dropArea.addEventListener(
//       eventName,
//       (e) => {
//         e.preventDefault();
//         e.stopPropagation();
//       },
//       false,
//     );
//   });

//   // Highlight drop area
//   ["dragenter", "dragover"].forEach((eventName) => {
//     dropArea.addEventListener(eventName, () => {
//       dropArea.classList.add("highlight");
//     });
//   });

//   // Remove highlight
//   ["dragleave", "drop"].forEach((eventName) => {
//     dropArea.addEventListener(eventName, () => {
//       dropArea.classList.remove("highlight");
//     });
//   });

//   // Handle dropped file
//   dropArea.addEventListener("drop", (e) => {
//     const file = e.dataTransfer.files[0];

//     if (file) {
//       handleSelectedFile(file);
//     }
//   });
// }

// // =========================================================
// // 5. HANDLE SELECTED FILE
// // =========================================================

// function handleSelectedFile(file) {
//   selectedFile = file;

//   // -------------------------------------------------------
//   // File Name
//   // -------------------------------------------------------

//   if (fileNameEl) {
//     fileNameEl.textContent = file.name;
//   }

//   // -------------------------------------------------------
//   // Set Audio Player
//   // -------------------------------------------------------

//   if (originalAudioPlayer) {
//     const audioURL = URL.createObjectURL(file);

//     originalAudioPlayer.src = audioURL;
//   }

//   // -------------------------------------------------------
//   // Reset Previous Prediction
//   // -------------------------------------------------------

//   if (emotionResultEl) {
//     emotionResultEl.textContent = "---";
//   }

//   if (probabilitiesEl) {
//     probabilitiesEl.innerHTML = "";
//   }

//   // -------------------------------------------------------
//   // Reset Previous Features
//   // -------------------------------------------------------

//   currentExtractedFeatures = null;
//   currentFeatureVector = null;

//   if (featureTableEl) {
//     featureTableEl.innerHTML = "";
//   }

//   // -------------------------------------------------------
//   // Reset Feature Status
//   // -------------------------------------------------------

//   const featureStatusText = document.getElementById("featureStatusText");
//   if (featureStatusText) {
//     featureStatusText.textContent =
//       "Please extract features to view the acoustic feature table.";
//   }
//   // if (featureStatusText) {
//   //   featureStatusText.textContent =
//   //     "Please classify the audio to view extracted features.";
//   // }

//   // -------------------------------------------------------
//   // Disable Classification While Loading
//   // -------------------------------------------------------

//   if (extractFeaturesBtn) {
//     extractFeaturesBtn.disabled = true;
//   }

//   if (classifyBtn) {
//     classifyBtn.disabled = true;
//   }

//   // -------------------------------------------------------
//   // Read Audio
//   // -------------------------------------------------------

//   const AudioContext = window.AudioContext || window.webkitAudioContext;

//   const audioContext = new AudioContext();

//   const reader = new FileReader();

//   reader.onload = function (e) {
//     audioContext
//       .decodeAudioData(e.target.result)

//       .then((buffer) => {
//         // -----------------------------------------------
//         // Audio Information
//         // -----------------------------------------------

//         if (durationEl) {
//           durationEl.textContent = `${buffer.duration.toFixed(2)} seconds`;
//         }

//         if (sampleRateEl) {
//           sampleRateEl.textContent = `${buffer.sampleRate} Hz`;
//         }

//         // -----------------------------------------------
//         // Create Waveforms
//         // -----------------------------------------------

//         createWaveforms(buffer);

//         // -----------------------------------------------
//         // Enable Classification
//         // -----------------------------------------------
//         // Enable Extract Features
//         if (extractFeaturesBtn) {
//           extractFeaturesBtn.disabled = false;
//         }

//         // Keep Classify disabled
//         if (classifyBtn) {
//           classifyBtn.disabled = true;
//         }

//         // Close AudioContext
//         if (audioContext.state !== "closed") {
//           audioContext.close();
//         }
//       })

//       .catch((error) => {
//         console.error("Audio decoding error:", error);

//         if (durationEl) {
//           durationEl.textContent = "Unknown";
//         }

//         if (sampleRateEl) {
//           sampleRateEl.textContent = "Unknown";
//         }
//         if (extractFeaturesBtn) {
//           extractFeaturesBtn.disabled = false;
//         }

//         if (classifyBtn) {
//           classifyBtn.disabled = false;
//         }

//         if (audioContext.state !== "closed") {
//           audioContext.close();
//         }
//       });
//   };

//   reader.readAsArrayBuffer(file);
// }

// // =========================================================
// // 6. CREATE ORIGINAL + PRE-EMPHASIZED WAVEFORMS
// // =========================================================

// function createWaveforms(buffer) {
//   if (!originalWaveformCanvas || !preEmphasisWaveformCanvas) {
//     console.error("Waveform canvas not found.");

//     return;
//   }

//   // -------------------------------------------------------
//   // Get Original Audio Samples
//   // -------------------------------------------------------

//   const channelData = buffer.getChannelData(0);

//   const sampleRate = buffer.sampleRate;

//   const totalSamples = channelData.length;

//   // -------------------------------------------------------
//   // APPLY PRE-EMPHASIS
//   //
//   // y[n] = x[n] - alpha*x[n-1]
//   //
//   // alpha = 0.97
//   // -------------------------------------------------------

//   const preEmphasizedData = new Float32Array(totalSamples);

//   // First sample
//   if (totalSamples > 0) {
//     preEmphasizedData[0] = channelData[0];
//   }

//   // Remaining samples
//   for (let i = 1; i < totalSamples; i++) {
//     preEmphasizedData[i] = channelData[i] - PRE_EMPHASIS * channelData[i - 1];
//   }

//   // -------------------------------------------------------
//   // DOWN-SAMPLE ONLY FOR DISPLAY
//   //
//   // IMPORTANT:
//   // Pre-emphasis was calculated on ALL samples.
//   // Down-sampling is only for chart display.
//   // -------------------------------------------------------

//   const maxPoints = 2000;

//   const step = Math.max(1, Math.ceil(totalSamples / maxPoints));

//   const originalData = [];
//   const preEmphasisData = [];

//   for (let i = 0; i < totalSamples; i += step) {
//     const time = i / sampleRate;

//     originalData.push({
//       x: time,
//       y: channelData[i],
//     });

//     preEmphasisData.push({
//       x: time,
//       y: preEmphasizedData[i],
//     });
//   }

//   // -------------------------------------------------------
//   // SHOW WAVEFORM SECTION
//   // -------------------------------------------------------

//   if (waveformSection) {
//     waveformSection.style.display = "block";
//   }

//   // -------------------------------------------------------
//   // DESTROY OLD CHARTS
//   // -------------------------------------------------------

//   if (originalWaveformChart) {
//     originalWaveformChart.destroy();

//     originalWaveformChart = null;
//   }

//   if (preEmphasisWaveformChart) {
//     preEmphasisWaveformChart.destroy();

//     preEmphasisWaveformChart = null;
//   }

//   // -------------------------------------------------------
//   // COMMON CHART OPTIONS
//   // -------------------------------------------------------

//   const commonOptions = {
//     responsive: true,

//     maintainAspectRatio: false,

//     animation: false,

//     interaction: {
//       intersect: false,

//       mode: "nearest",
//     },

//     scales: {
//       x: {
//         type: "linear",
//         min: 0,
//         max: buffer.duration,
//         title: {
//           display: true,
//           text: "Time (seconds)",
//         },

//         ticks: {
//           stepSize: 1, //0.5

//           callback: function (value) {
//             return Number(value).toFixed(0); //toFixed(1)
//           },

//           maxTicksLimit: 7,
//         },
//       },

//       y: {
//         // min: -1,
//         // max: 1

//         title: {
//           display: true,

//           text: "Amplitude",
//         },
//       },
//     },

//     plugins: {
//       legend: {
//         display: false,
//       },

//       tooltip: {
//         callbacks: {
//           title: function (tooltipItems) {
//             const x = tooltipItems[0].parsed.x;

//             return "Time: " + x.toFixed(3) + " seconds";
//           },

//           label: function (context) {
//             return "Amplitude: " + context.parsed.y.toFixed(4);
//           },
//         },
//       },
//     },
//   };

//   // =======================================================
//   // ORIGINAL WAVEFORM
//   // =======================================================

//   originalWaveformChart = new Chart(originalWaveformCanvas, {
//     type: "line",

//     data: {
//       datasets: [
//         {
//           label: "Original Waveform",

//           data: originalData,

//           borderWidth: 1,

//           pointRadius: 0,

//           tension: 0,

//           fill: false,
//         },
//       ],
//     },

//     options: commonOptions,
//   });

//   // =======================================================
//   // PRE-EMPHASIZED WAVEFORM
//   // =======================================================

//   preEmphasisWaveformChart = new Chart(preEmphasisWaveformCanvas, {
//     type: "line",

//     data: {
//       datasets: [
//         {
//           label: "Pre-emphasized Waveform",

//           data: preEmphasisData,

//           borderWidth: 1,

//           pointRadius: 0,

//           tension: 0,

//           fill: false,
//         },
//       ],
//     },

//     options: commonOptions,
//   });
// }
// // =========================================================
// // =========================================================
// // 7. EXTRACT ACOUSTIC FEATURES
// // =========================================================

// if (extractFeaturesBtn) {
//   extractFeaturesBtn.addEventListener("click", async () => {
//     // No file selected
//     if (!selectedFile) {
//       alert("Please upload an audio file first.");
//       return;
//     }

//     // Disable Extract button
//     extractFeaturesBtn.disabled = true;

//     extractFeaturesBtn.textContent = "⏳ Extracting...";

//     // Clear previous classification result
//     if (emotionResultEl) {
//       emotionResultEl.textContent = "---";
//     }

//     if (probabilitiesEl) {
//       probabilitiesEl.innerHTML = "";
//     }

//     // Clear previous feature data
//     currentExtractedFeatures = null;
//     currentFeatureVector = null;

//     if (featureTableEl) {
//       featureTableEl.innerHTML = "";
//     }

//     const featureStatusText = document.getElementById("featureStatusText");

//     if (featureStatusText) {
//       featureStatusText.textContent =
//         "Extracting MFCC, Chroma, and Mel-Spectrogram features...";
//     }

//     // -----------------------------------------------------
//     // Form Data
//     // -----------------------------------------------------

//     const formData = new FormData();

//     formData.append("audio", selectedFile);

//     try {
//       // ---------------------------------------------------
//       // Send Audio to Flask
//       // ---------------------------------------------------

//       const response = await fetch("/extract_features", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();

//       // ---------------------------------------------------
//       // Successful Extraction
//       // ---------------------------------------------------

//       if (response.ok && data.success) {
//         currentExtractedFeatures = data.extracted_features;

//         currentFeatureVector = data.feature_vector;

//         // Show feature table
//         renderFeatureTableByTab(activeFeatureTab);

//         // Enable Classification
//         if (classifyBtn) {
//           classifyBtn.disabled = false;
//         }

//         if (featureStatusText) {
//           featureStatusText.textContent =
//             "Features extracted successfully. You can now classify the emotion.";
//         }
//       } else {
//         alert(`Error: ${data.error || "Failed to extract features"}`);

//         if (featureStatusText) {
//           featureStatusText.textContent = "Feature extraction failed.";
//         }
//       }
//     } catch (error) {
//       console.error("Feature Extraction API Error:", error);

//       alert("Server Error during feature extraction.");

//       if (featureStatusText) {
//         featureStatusText.textContent = "Feature extraction failed.";
//       }
//     } finally {
//       // Restore Extract button
//       extractFeaturesBtn.disabled = false;

//       extractFeaturesBtn.textContent = "📊 Extract Features";
//     }
//   });
// }

// // =========================================================
// // 7. CLASSIFY EMOTION
// // =========================================================

// // if (classifyBtn) {
// //   classifyBtn.addEventListener("click", async () => {
// //     // No file selected
// //     if (!selectedFile) {
// //       return;
// //     }

// //     // Disable button
// //     classifyBtn.disabled = true;

// //     classifyBtn.textContent = "⏳ Classifying...";

// //     // Show processing state
// //     if (emotionResultEl) {
// //       emotionResultEl.textContent = "Analyzing...";
// //     }

// //     if (probabilitiesEl) {
// //       probabilitiesEl.innerHTML = "<p>Processing features...</p>";
// //     }

// //     // Clear previous features
// //     currentExtractedFeatures = null;

// //     if (featureTableEl) {
// //       featureTableEl.innerHTML = "";
// //     }

// //     // -----------------------------------------------------
// //     // Form Data
// //     // -----------------------------------------------------

// //     const formData = new FormData();

// //     formData.append("audio", selectedFile);

// //     try {
// //       // ---------------------------------------------------
// //       // Send Audio to Flask
// //       // ---------------------------------------------------

// //       const response = await fetch("/predict", {
// //         method: "POST",

// //         body: formData,
// //       });

// //       // ---------------------------------------------------
// //       // Read JSON Response
// //       // ---------------------------------------------------

// //       const data = await response.json();

// //       // ---------------------------------------------------
// //       // Successful Prediction
// //       // ---------------------------------------------------

// //       if (response.ok && data.success) {
// //         // -----------------------------------------------
// //         // Prediction
// //         // -----------------------------------------------

// //         displayPredictionResult(data.emotion, data.confidence);

// //         // -----------------------------------------------
// //         // Probabilities
// //         // -----------------------------------------------

// //         if (data.probabilities) {
// //           displayProbabilities(data.probabilities);
// //         }

// //         // -----------------------------------------------
// //         // Extracted Features
// //         // -----------------------------------------------

// //         if (data.extracted_features) {
// //           currentExtractedFeatures = data.extracted_features;

// //           renderFeatureTableByTab(activeFeatureTab);
// //         }
// //       }

// //       // ---------------------------------------------------
// //       // Error Response
// //       // ---------------------------------------------------
// //       else {
// //         alert(`Error: ${data.error || "Failed to process audio"}`);

// //         if (emotionResultEl) {
// //           emotionResultEl.textContent = "Error";
// //         }
// //       }
// //     } catch (error) {
// //       // -----------------------------------------------------
// //       // Server / Network Error
// //       // -----------------------------------------------------

// //       console.error("API Error:", error);

// //       alert("Server Error!");

// //       if (emotionResultEl) {
// //         emotionResultEl.textContent = "Error";
// //       }
// //     } finally {
// //       // -----------------------------------------------------
// //       // Restore Button
// //       // -----------------------------------------------------

// //       classifyBtn.disabled = false;

// //       classifyBtn.textContent = "🎯 Classify Emotion";
// //     }
// //   });
// // }
// // =========================================================
// // 8. CLASSIFY EMOTION
// // =========================================================
// // =========================================================
// // EXTRACT FEATURES
// // =========================================================

// if (extractFeaturesBtn) {
//   extractFeaturesBtn.addEventListener("click", async () => {
//     if (!selectedFile) {
//       return;
//     }

//     extractFeaturesBtn.disabled = true;
//     extractFeaturesBtn.textContent = "⏳ Extracting...";

//     const featureStatusText = document.getElementById("featureStatusText");

//     if (featureStatusText) {
//       featureStatusText.textContent = "Extracting acoustic features...";
//     }

//     const formData = new FormData();
//     formData.append("audio", selectedFile);

//     try {
//       const response = await fetch("/extract_features", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();

//       if (response.ok && data.success) {
//         // Save extracted features
//         currentExtractedFeatures = data.extracted_features;

//         // Render feature table
//         renderFeatureTableByTab(activeFeatureTab);

//         // Enable Classify
//         if (classifyBtn) {
//           classifyBtn.disabled = false;
//         }

//         if (featureStatusText) {
//           featureStatusText.textContent =
//             "Acoustic features extracted successfully.";
//         }
//       } else {
//         alert(
//           "Feature Extraction Error: " +
//             (data.error || "Failed to extract features."),
//         );
//       }
//     } catch (error) {
//       console.error("Feature extraction error:", error);

//       alert("Server Error while extracting features.");
//     } finally {
//       extractFeaturesBtn.disabled = false;
//       extractFeaturesBtn.textContent = "📊 Extract Features";
//     }
//   });
// }
// if (classifyBtn) {
//   classifyBtn.addEventListener("click", async () => {
//     // -----------------------------------------------------
//     // Check File
//     // -----------------------------------------------------

//     if (!selectedFile) {
//       alert("Please upload an audio file first.");
//       return;
//     }

//     // -----------------------------------------------------
//     // Check Extracted Features
//     // -----------------------------------------------------

//     if (!currentFeatureVector) {
//       alert("Please extract features before classification.");
//       return;
//     }

//     // -----------------------------------------------------
//     // Disable Button
//     // -----------------------------------------------------

//     classifyBtn.disabled = true;

//     if (extractFeaturesBtn) {
//       extractFeaturesBtn.disabled = true;
//     }

//     classifyBtn.textContent = "⏳ Classifying...";

//     // -----------------------------------------------------
//     // Show Processing State
//     // -----------------------------------------------------

//     if (emotionResultEl) {
//       emotionResultEl.textContent = "Analyzing...";
//     }

//     if (probabilitiesEl) {
//       probabilitiesEl.innerHTML = "<p>Processing classification...</p>";
//     }

//     // -----------------------------------------------------
//     // Form Data
//     // -----------------------------------------------------

//     const formData = new FormData();

//     formData.append("audio", selectedFile);

//     formData.append("feature_vector", JSON.stringify(currentFeatureVector));

//     try {
//       // ---------------------------------------------------
//       // Send Extracted Features to Flask
//       // ---------------------------------------------------

//       const response = await fetch("/predict", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();

//       // ---------------------------------------------------
//       // Successful Prediction
//       // ---------------------------------------------------

//       if (response.ok && data.success) {
//         // Prediction
//         displayPredictionResult(data.emotion, data.confidence);

//         // Probabilities
//         if (data.probabilities) {
//           displayProbabilities(data.probabilities);
//         }

//         // Keep feature table
//         if (data.extracted_features) {
//           currentExtractedFeatures = data.extracted_features;

//           renderFeatureTableByTab(activeFeatureTab);
//         }

//         const featureStatusText = document.getElementById("featureStatusText");

//         if (featureStatusText) {
//           featureStatusText.textContent =
//             "Features extracted successfully. Classification completed.";
//         }
//       } else {
//         alert(`Error: ${data.error || "Failed to classify audio"}`);

//         if (emotionResultEl) {
//           emotionResultEl.textContent = "Error";
//         }
//       }
//     } catch (error) {
//       console.error("Prediction API Error:", error);

//       alert("Server Error during classification.");

//       if (emotionResultEl) {
//         emotionResultEl.textContent = "Error";
//       }
//     } finally {
//       // ---------------------------------------------------
//       // Restore Buttons
//       // ---------------------------------------------------

//       classifyBtn.disabled = false;

//       if (extractFeaturesBtn) {
//         extractFeaturesBtn.disabled = false;
//       }

//       classifyBtn.textContent = "🎯 Classify Emotion";
//     }
//   });
// }
// // =========================================================
// // 8. DISPLAY PREDICTION RESULT
// // =========================================================

// function displayPredictionResult(emotion, confidence) {
//   if (!emotionResultEl) {
//     return;
//   }

//   emotionResultEl.innerHTML = `

//     <span
//       style="
//         color: #2b5cff;
//         text-transform: capitalize;
//       "
//     >
//       ${emotion}
//     </span>


//     <div
//       style="
//         font-size: 0.9rem;
//         font-weight: normal;
//         color: #666;
//         margin-top: 5px;
//       "
//     >

//       Confidence:

//       <strong>
//         ${Number(confidence).toFixed(2)}%
//       </strong>

//     </div>

//   `;
// }

// // =========================================================
// // 9. DISPLAY PROBABILITIES
// // =========================================================

// function displayProbabilities(probabilities) {
//   if (!probabilitiesEl) {
//     return;
//   }

//   probabilitiesEl.innerHTML = "";

//   // Sort from highest to lowest
//   const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

//   sorted.forEach(([emotion, prob]) => {
//     const item = document.createElement("div");

//     item.style.marginBottom = "10px";

//     item.innerHTML = `

//         <div
//           style="
//             display: flex;
//             justify-content: space-between;
//             font-size: 0.85rem;
//             margin-bottom: 3px;
//           "
//         >

//           <span
//             style="
//               text-transform: capitalize;
//             "
//           >
//             ${emotion}
//           </span>


//           <span>
//             ${Number(prob).toFixed(2)}%
//           </span>

//         </div>


//         <div
//           style="
//             background-color: #e2e8f0;
//             border-radius: 6px;
//             height: 8px;
//             overflow: hidden;
//           "
//         >

//           <div
//             style="
//               width: ${Number(prob)}%;
//               background-color: #2b5cff;
//               height: 100%;
//               transition: width 0.5s ease;
//             "
//           ></div>

//         </div>

//       `;

//     probabilitiesEl.appendChild(item);
//   });
// }

// // =========================================================
// // 10. FEATURE TAB SWITCHING
// // =========================================================

// function switchFeatureTab(tabName) {
//   activeFeatureTab = tabName;

//   // -------------------------------------------------------
//   // Remove active state from all buttons
//   // -------------------------------------------------------

//   document.querySelectorAll(".gui-tab-btn").forEach((btn) => {
//     btn.classList.remove("active");
//   });

//   // -------------------------------------------------------
//   // Add active state to selected button
//   // -------------------------------------------------------

//   const activeBtn = document.getElementById(`btn${tabName}`);

//   if (activeBtn) {
//     activeBtn.classList.add("active");
//   }

//   // -------------------------------------------------------
//   // Render selected feature table
//   // -------------------------------------------------------

//   renderFeatureTableByTab(tabName);
// }

// // =========================================================
// // 11. RENDER FEATURE TABLE BY TAB
// // =========================================================

// function renderFeatureTableByTab(tabName) {
//   const table = document.getElementById("featureTable");

//   const featureStatusText = document.getElementById("featureStatusText");

//   if (!table || !featureStatusText) {
//     return;
//   }

//   // Clear table
//   table.innerHTML = "";

//   // =======================================================
//   // NO FEATURES YET
//   // =======================================================

//   if (!currentExtractedFeatures) {
//     featureStatusText.textContent =
//       "No extracted features. Please upload an audio file and click Extract Features.";
//     // "No extracted features. Please upload and classify an audio file.";

//     return;
//   }

//   // =======================================================
//   // SELECT FEATURE PREFIX
//   // =======================================================

//   let prefix = "";
//   let expectedCount = 0;
//   let displayName = "";

//   if (tabName === "MFCC") {
//     prefix = "MFCC_";

//     expectedCount = 80;

//     displayName = "MFCC";
//   } else if (tabName === "Chroma") {
//     prefix = "Chroma_";

//     expectedCount = 24;

//     displayName = "CHROMA";
//   } else if (tabName === "Mel") {
//     prefix = "Mel_";

//     expectedCount = 256;

//     displayName = "MEL SPECTROGRAM";
//   }

//   // =======================================================
//   // FILTER FEATURES
//   // =======================================================

//   // const filteredKeys =
//   //   Object.keys(
//   //     currentExtractedFeatures
//   //   )
//   //     .filter(
//   //       (key) =>
//   //         key.startsWith(prefix)
//   //     );
//   const filteredKeys = Object.keys(currentExtractedFeatures)
//     .filter((key) => key.startsWith(prefix))
//     .sort((a, b) => {
//       // -----------------------------------------------
//       // Get feature number
//       // Example:
//       // Chroma_1_mean  → 1
//       // Chroma_12_std  → 12
//       // -----------------------------------------------

//       const numberA = parseInt(a.match(/_(\d+)_/)?.[1] || "0");

//       const numberB = parseInt(b.match(/_(\d+)_/)?.[1] || "0");

//       // -----------------------------------------------
//       // First sort by feature number
//       // -----------------------------------------------

//       if (numberA !== numberB) {
//         return numberA - numberB;
//       }

//       // -----------------------------------------------
//       // Same number:
//       // Mean first
//       // Std second
//       // -----------------------------------------------

//       const typeA = a.endsWith("_mean") ? 0 : 1;

//       const typeB = b.endsWith("_mean") ? 0 : 1;

//       return typeA - typeB;
//     });

//   // =======================================================
//   // STATUS TEXT
//   // =======================================================

//   featureStatusText.textContent = `Showing ${filteredKeys.length} ${displayName} features`;

//   // =======================================================
//   // DISPLAY FEATURES
//   // =======================================================

//   filteredKeys.forEach((key) => {
//     const rawValue = currentExtractedFeatures[key];

//     const value =
//       rawValue !== undefined &&
//       rawValue !== null &&
//       !Number.isNaN(Number(rawValue))
//         ? Number(rawValue).toFixed(4)
//         : "-";

//     const type = key.endsWith("_std") ? "Std" : "Mean";

//     const row = document.createElement("tr");

//     row.innerHTML = `

//         <td>
//           <strong>
//             ${key}
//           </strong>
//         </td>


//         <td>
//           ${type}
//         </td>


//         <td
//           style="
//             text-align: right;
//             font-family: monospace;
//             color: #1f6bba;
//             font-weight: bold;
//           "
//         >
//           ${value}
//         </td>

//       `;

//     table.appendChild(row);
//   });
// }

// // =========================================================
// // 12. CLEAR ALL
// // =========================================================

// function clearAll() {
//   // -------------------------------------------------------
//   // Clear Selected File
//   // -------------------------------------------------------

//   selectedFile = null;

//   // -------------------------------------------------------
//   // Clear Feature Data
//   // -------------------------------------------------------

//   currentExtractedFeatures = null;

//   activeFeatureTab = "MFCC";

//   // -------------------------------------------------------
//   // Clear File Input
//   // -------------------------------------------------------

//   if (audioFileInput) {
//     audioFileInput.value = "";
//   }

//   // -------------------------------------------------------
//   // Clear Audio Player
//   // -------------------------------------------------------

//   if (originalAudioPlayer) {
//     originalAudioPlayer.pause();

//     originalAudioPlayer.src = "";

//     originalAudioPlayer.load();
//   }

//   // -------------------------------------------------------
//   // Destroy Original Waveform
//   // -------------------------------------------------------

//   if (originalWaveformChart) {
//     originalWaveformChart.destroy();

//     originalWaveformChart = null;
//   }

//   // -------------------------------------------------------
//   // Destroy Pre-emphasis Waveform
//   // -------------------------------------------------------

//   if (preEmphasisWaveformChart) {
//     preEmphasisWaveformChart.destroy();

//     preEmphasisWaveformChart = null;
//   }

//   // -------------------------------------------------------
//   // Hide Waveform Section
//   // -------------------------------------------------------

//   if (waveformSection) {
//     waveformSection.style.display = "none";
//   }

//   // -------------------------------------------------------
//   // Clear Audio Information
//   // -------------------------------------------------------

//   if (fileNameEl) {
//     fileNameEl.textContent = "No file selected";
//   }

//   if (durationEl) {
//     durationEl.textContent = "-";
//   }

//   if (sampleRateEl) {
//     sampleRateEl.textContent = "-";
//   }

//   // -------------------------------------------------------
//   // Clear Prediction
//   // -------------------------------------------------------

//   if (emotionResultEl) {
//     emotionResultEl.textContent = "---";
//   }

//   if (probabilitiesEl) {
//     probabilitiesEl.innerHTML = "";
//   }

//   // -------------------------------------------------------
//   // Clear Feature Table
//   // -------------------------------------------------------

//   if (featureTableEl) {
//     featureTableEl.innerHTML = "";
//   }

//   // -------------------------------------------------------
//   // Reset Feature Status
//   // -------------------------------------------------------

//   const featureStatusText = document.getElementById("featureStatusText");

//   if (featureStatusText) {
//     featureStatusText.textContent =
//       "Please upload an audio file to view extracted features.";
//   }

//   // -------------------------------------------------------
//   // Reset Active Tab
//   // -------------------------------------------------------

//   document.querySelectorAll(".gui-tab-btn").forEach((btn) => {
//     btn.classList.remove("active");
//   });

//   const defaultTab = document.getElementById("btnMFCC");

//   if (defaultTab) {
//     defaultTab.classList.add("active");
//   }

//   // -------------------------------------------------------
//   // Disable Classify Button
//   // -------------------------------------------------------
//   if (extractFeaturesBtn) {
//     extractFeaturesBtn.disabled = true;
//     extractFeaturesBtn.textContent = "📊 Extract Features";
//   }
//   if (classifyBtn) {
//     classifyBtn.disabled = true;
//     classifyBtn.textContent = "🎯 Classify Emotion";
//   }
// }

// // =========================================================
// // 13. PAGE LOAD
// // =========================================================

// document.addEventListener("DOMContentLoaded", () => {
//   // Do NOT render feature data on page load.
//   // Feature table will appear after classification.

//   currentExtractedFeatures = null;
//   currentFeatureVector = null;

//   activeFeatureTab = "MFCC";

//   if (extractFeaturesBtn) {
//     extractFeaturesBtn.disabled = true;
//   }

//   if (classifyBtn) {
//     classifyBtn.disabled = true;
//   }

//   // Set MFCC tab active
//   const defaultTab = document.getElementById("btnMFCC");

//   if (defaultTab) {
//     defaultTab.classList.add("active");
//   }
// });
