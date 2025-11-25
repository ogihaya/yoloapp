(() => {
    const script = document.currentScript;
    const RUN_ENDPOINT = script?.dataset.runEndpoint || "";
    const MAX_IMAGES = Number(script?.dataset.maxImages) || 24;

    const state = {
        model: null,
        images: [],
        settings: {
            img_size: 640,
            num_workers: 4,
            min_confidence: 0.25,
            min_iou: 0.45,
            max_bbox: 300,
        },
        logs: [],
        classNames: [],
        classEditorText: "",
        lastStats: null,
        lastDevice: "",
        isRunning: false,
    };

    const INFERENCE_CLASS_STORAGE_KEY = "yoloApp.prefillInferenceClasses";

    const elements = {
        toast: document.getElementById("toast"),
        modelInput: document.getElementById("modelInput"),
        modelDropzone: document.getElementById("modelDropzone"),
        modelStatus: document.getElementById("modelStatus"),
        modelInfo: document.getElementById("modelInfo"),
        imageInput: document.getElementById("imageInput"),
        imageDropzone: document.getElementById("imageDropzone"),
        imageGrid: document.getElementById("imageGrid"),
        imageCounter: document.getElementById("imageCounter"),
        settingsForm: document.getElementById("settingsForm"),
        runLog: document.getElementById("runLog"),
        runBtn: document.getElementById("runInference"),
        resetBtn: document.getElementById("resetWorkspace"),
        classEditor: document.getElementById("classEditor"),
        classPreview: document.getElementById("classPreview"),
    };

    let toastTimer = null;

    const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : `id-${Date.now()}-${Math.random()}`);

    const formatBytes = (bytes) => {
        if (!Number.isFinite(bytes)) return "-";
        if (bytes === 0) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
        return `${(bytes / 1024 ** index).toFixed(1)} ${units[index]}`;
    };

    const formatPercent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

    const showToast = (message, isError = false) => {
        elements.toast.textContent = message;
        elements.toast.classList.toggle("is-error", Boolean(isError));
        elements.toast.classList.add("is-visible");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 2800);
    };

    const restorePrefilledClasses = () => {
        const raw = sessionStorage.getItem(INFERENCE_CLASS_STORAGE_KEY);
        if (!raw) return;
        sessionStorage.removeItem(INFERENCE_CLASS_STORAGE_KEY);
        try {
            const payload = JSON.parse(raw);
            const incoming = Array.isArray(payload?.classes) ? payload.classes : [];
            const labels = incoming
                .map((cls, idx) => {
                    const label = typeof cls?.label === "string" ? cls.label.trim() : "";
                    return label || `class_${idx}`;
                })
                .filter(Boolean);
            if (!labels.length) return;
            state.classNames = labels;
            state.classEditorText = labels.join("\n");
            showToast("Val画面のクラス設定を読み込みました");
        } catch (error) {
            console.error("Failed to restore classes for inference page", error);
        }
    };

    const getManualClasses = () =>
        state.classEditorText
            .split(/\r?\n/)
            .map((name) => name.trim())
            .filter((name, index, arr) => name && arr.indexOf(name) === index);

    const renderClassPreview = () => {
        const manualClasses = getManualClasses();
        const previewClasses = manualClasses.length ? manualClasses : state.classNames;
        elements.classPreview.innerHTML = "";
        if (!previewClasses.length) {
            const empty = document.createElement("p");
            empty.className = "class-helper";
            empty.textContent = "クラス名を入力するか、一度推論を実行するとここに一覧が表示されます。";
            elements.classPreview.appendChild(empty);
            return;
        }
        previewClasses.forEach((name, idx) => {
            const chip = document.createElement("span");
            chip.className = "class-chip";
            chip.textContent = `${idx}: ${name}`;
            elements.classPreview.appendChild(chip);
        });
        if (!manualClasses.length && state.classNames.length) {
            const helper = document.createElement("p");
            helper.className = "class-helper";
            helper.textContent = "※推論結果から取得したクラスを表示しています。";
            elements.classPreview.appendChild(helper);
        }
    };

    const updateSettings = () => {
        const formData = new FormData(elements.settingsForm);
        Object.keys(state.settings).forEach((key) => {
            const value = formData.get(key);
            if (value !== null) {
                const num = Number(value);
                state.settings[key] = Number.isFinite(num) ? num : value;
            }
        });
    };

    const renderModelInfo = () => {
        const { modelStatus, modelInfo } = elements;
        if (!state.model) {
            modelStatus.textContent = "モデル未読込";
            modelStatus.className = "status-chip status-idle";
            modelInfo.classList.remove("is-visible");
            modelInfo.innerHTML = "";
            return;
        }

        modelStatus.textContent = state.isRunning ? "モデル使用中" : "モデル読込済み";
        modelStatus.className = `status-chip ${state.isRunning ? "status-running" : "status-ready"}`;
        const { name, size, importedAt } = state.model;
        const manualClasses = getManualClasses();
        const classesForDisplay = manualClasses.length ? manualClasses : state.classNames;
        const classesInfo = classesForDisplay.length
            ? `<p style="margin:0.2rem 0 0;color:var(--muted);">検出クラス: ${classesForDisplay.join(", ")}</p>`
            : "";
        const stats = state.lastStats;
        const statsInfo = stats
            ? `<p style="margin:0.2rem 0 0;color:var(--muted);">前回: ${stats.total_images ?? 0}枚 / ${
                  stats.total_detections ?? 0
              }件 / ${(Number(stats.total_time_ms ?? 0)).toFixed(1)}ms${state.lastDevice ? ` / Device: ${state.lastDevice}` : ""}</p>`
            : "";
        modelInfo.classList.add("is-visible");
        modelInfo.innerHTML = `
            <strong>${name}</strong>
            <p style="margin:0.4rem 0 0.2rem;">サイズ: ${formatBytes(size)}</p>
            <p style="margin:0;color:var(--muted);">読込時刻: ${importedAt.toLocaleString()}</p>
            ${classesInfo}
            ${statsInfo}
        `;
    };

    const renderLogs = () => {
        if (!state.logs.length) {
            elements.runLog.innerHTML = '<li style="color:var(--muted);">推論履歴はまだありません。</li>';
            return;
        }
        elements.runLog.innerHTML = state.logs
            .map(
                (log) =>
                    `<li><span class="log-title">${log.title}</span><span class="log-meta">${log.meta}</span></li>`,
            )
            .join("");
    };

    const getStatusClass = (status) => {
        switch (status) {
            case "running":
                return "status-running";
            case "completed":
                return "status-ready";
            case "error":
                return "status-error";
            default:
                return "status-idle";
        }
    };

    const getStatusLabel = (image) => {
        if (image.status === "running") return "推論中";
        if (image.status === "completed") {
            return image.numDetections ? `${image.numDetections}件検出` : "検出なし";
        }
        if (image.status === "error") return "エラー";
        return "未推論";
    };

    const getOverlayText = (status) => {
        if (status === "running") return "推論中...";
        if (status === "error") return "推論に失敗しました";
        if (status === "pending") return "推論待ち";
        return "";
    };

    const renderImageGrid = () => {
        elements.imageGrid.innerHTML = "";
        elements.imageCounter.textContent = `${state.images.length}枚`;
        if (!state.images.length) {
            const placeholder = document.createElement("div");
            placeholder.className = "image-card";
            placeholder.innerHTML = '<p class="result-empty">画像を登録するとここに一覧が表示されます。</p>';
            elements.imageGrid.appendChild(placeholder);
            return;
        }

        state.images.forEach((image) => {
            const card = document.createElement("article");
            card.className = "image-card";

            const header = document.createElement("header");
            const title = document.createElement("div");
            title.innerHTML = `<strong>${image.name}</strong><br/><span style="color:var(--muted)">${formatBytes(
                image.size,
            )}</span>`;
            const chip = document.createElement("span");
            chip.className = `status-chip ${getStatusClass(image.status)}`;
            chip.textContent = getStatusLabel(image);
            header.appendChild(title);
            header.appendChild(chip);
            card.appendChild(header);

            const preview = document.createElement("div");
            preview.className = "image-preview";
            if (image.status === "pending") preview.classList.add("is-pending");
            if (image.status === "running") preview.classList.add("is-running");
            if (image.status === "error") preview.classList.add("is-error");
            const imgEl = document.createElement("img");
            imgEl.src = image.resultImage || image.previewSrc;
            imgEl.alt = image.name;
            preview.appendChild(imgEl);
            const overlay = document.createElement("div");
            overlay.className = "preview-overlay";
            overlay.textContent = getOverlayText(image.status);
            preview.appendChild(overlay);
            card.appendChild(preview);

            let bodyContent = "";
            if (image.status === "error") {
                bodyContent = `<p class="result-error">${image.error || "推論に失敗しました"}</p>`;
            } else if (image.results?.length) {
                const rows = image.results
                    .map((result) => {
                        const bbox = result.bbox || {};
                        return `<li>
                            <span class="result-chip">${result.class_name ?? result.class_id}</span>
                            <span class="result-meta">${formatPercent(result.confidence ?? 0)} / (${Math.round(
                            bbox.x1 ?? 0,
                        )}, ${Math.round(bbox.y1 ?? 0)})→(${Math.round(bbox.x2 ?? 0)}, ${Math.round(
                            bbox.y2 ?? 0,
                        )})</span>
                        </li>`;
                    })
                    .join("");
                bodyContent = `<ul class="result-list">${rows}</ul>`;
            } else if (image.status === "completed") {
                bodyContent = "<p class=\"result-empty\">検出は見つかりませんでした。</p>";
            } else {
                bodyContent = "<p class=\"result-empty\">推論待ちです。</p>";
            }
            const info = document.createElement("div");
            info.innerHTML = bodyContent;
            card.appendChild(info);

            const actions = document.createElement("div");
            actions.className = "card-actions";
            const downloadBtn = document.createElement("button");
            downloadBtn.className = "btn-outline";
            downloadBtn.textContent = "結果画像を保存";
            downloadBtn.disabled = !image.resultImage;
            downloadBtn.addEventListener("click", () => downloadImage(image));
            actions.appendChild(downloadBtn);
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn-outline btn-danger";
            deleteBtn.textContent = "画像を削除";
            deleteBtn.addEventListener("click", () => {
                if (window.confirm(`画像「${image.name}」を削除しますか？`)) {
                    removeImage(image.id);
                }
            });
            actions.appendChild(deleteBtn);
            card.appendChild(actions);

            elements.imageGrid.appendChild(card);
        });
    };

    const handleModelFiles = (files) => {
        if (!files.length) return;
        const file = files[0];
        if (!file.name.toLowerCase().endsWith(".pt")) {
            showToast("拡張子が .pt のモデルファイルを指定してください。", true);
            return;
        }
        state.model = {
            id: uuid(),
            name: file.name,
            size: file.size,
            file,
            importedAt: new Date(),
        };
        renderModelInfo();
        showToast("モデルファイルを読み込みました");
    };

    const handleImageFiles = (files) => {
        if (!files.length) return;
        const canAdd = Math.max(0, MAX_IMAGES - state.images.length);
        const candidates = Array.from(files)
            .filter((file) => file.type.startsWith("image/"))
            .slice(0, canAdd);
        if (!candidates.length) {
            showToast("画像ファイルを選択してください。", true);
            return;
        }
        candidates.forEach((file) => {
            const reader = new FileReader();
            reader.onload = (event) => {
                state.images.push({
                    id: uuid(),
                    name: file.name,
                    size: file.size,
                    file,
                    previewSrc: event.target.result,
                    status: "pending",
                    results: [],
                    resultImage: null,
                    error: null,
                    duration: null,
                    numDetections: 0,
                });
                renderImageGrid();
            };
            reader.readAsDataURL(file);
        });
        if (files.length > candidates.length) {
            showToast(`最大${MAX_IMAGES}枚まで登録できます。`, true);
        } else {
            showToast(`${candidates.length}件の画像を追加しました。`);
        }
    };

    const downloadImage = (image) => {
        if (!image.resultImage) {
            showToast("推論後の画像がまだありません。", true);
            return;
        }
        const link = document.createElement("a");
        link.href = image.resultImage;
        link.download = `result_${image.name}`;
        link.click();
    };

    const resetWorkspace = () => {
        state.images = [];
        state.logs = [];
        state.lastStats = null;
        renderImageGrid();
        renderLogs();
        renderModelInfo();
        showToast("画像と履歴をクリアしました");
    };

    const removeImage = (imageId) => {
        const target = state.images.find((img) => img.id === imageId);
        if (!target) return;
        state.images = state.images.filter((img) => img.id !== imageId);
        renderImageGrid();
        showToast(`画像「${target.name}」を削除しました`);
    };

    const applyInferenceResults = (payload) => {
        const resultMap = new Map();
        (payload.results || []).forEach((item) => {
            resultMap.set(item.image_id, item);
        });

        const incomingClasses = payload.class_names || [];
        if (incomingClasses.length) {
            state.classNames = incomingClasses;
            if (!state.classEditorText.trim()) {
                state.classEditorText = incomingClasses.join("\n");
                elements.classEditor.value = state.classEditorText;
            }
        }
        state.lastStats = payload.stats || null;
        state.lastDevice = payload.device || "";

        state.images.forEach((image) => {
            const result = resultMap.get(image.id);
            if (!result) {
                image.status = "error";
                image.error = "結果が取得できませんでした";
                image.resultImage = null;
                image.results = [];
                return;
            }
            image.status = "completed";
            image.results = result.detections || [];
            image.resultImage = result.result_image || image.resultImage || image.previewSrc;
            image.previewSrc = image.resultImage || image.previewSrc;
            image.numDetections = result.num_detections ?? image.results.length;
            image.duration = result.duration_ms ?? null;
            image.error = null;
        });

        const stats = payload.stats || {};
        const meta = `${new Date().toLocaleString()} / Device: ${payload.device || "N/A"} / 検出 ${
            stats.total_detections ?? 0
        }件 / ${(Number(stats.total_time_ms ?? 0)).toFixed(1)} ms`;
        state.logs.unshift({
            id: uuid(),
            title: `推論完了 (${stats.total_images ?? state.images.length}枚)`,
            meta,
        });
        if (state.logs.length > 30) state.logs.pop();
        renderClassPreview();
    };

    const executeInference = async () => {
        if (!state.model) {
            showToast("まずモデルファイルを読み込んでください。", true);
            return;
        }
        if (!state.images.length) {
            showToast("推論したい画像を追加してください。", true);
            return;
        }
        if (state.isRunning) return;
        if (!RUN_ENDPOINT) {
            showToast("推論APIエンドポイントが設定されていません。", true);
            return;
        }

        updateSettings();
        state.isRunning = true;
        elements.runBtn.disabled = true;
        state.images.forEach((image) => {
            image.status = "running";
            image.results = [];
            image.error = null;
            image.resultImage = image.resultImage || image.previewSrc;
        });
        renderImageGrid();
        renderModelInfo();

        const formData = new FormData();
        formData.append("model", state.model.file, state.model.name);
        Object.entries(state.settings).forEach(([key, value]) => {
            formData.append(key, value);
        });
        const manualClasses = getManualClasses();
        if (manualClasses.length) {
            formData.append("class_names", JSON.stringify(manualClasses));
        }
        formData.append(
            "image_metadata",
            JSON.stringify(state.images.map(({ id, name }) => ({ id, name }))),
        );
        state.images.forEach((image) => {
            formData.append("images", image.file, image.name);
        });

        try {
            const response = await fetch(RUN_ENDPOINT, { method: "POST", body: formData });
            const contentType = response.headers.get("content-type") || "";
            const payload = contentType.includes("application/json") ? await response.json() : null;
            if (!response.ok) {
                const message = payload?.error || "推論API呼び出しに失敗しました。";
                throw new Error(message);
            }
            applyInferenceResults(payload || {});
            renderLogs();
            renderModelInfo();
            renderImageGrid();
            showToast("推論が完了しました。");
        } catch (error) {
            state.images.forEach((image) => {
                image.status = "error";
                image.error = error.message || "推論中にエラーが発生しました";
            });
            showToast(error.message || "推論に失敗しました。", true);
            renderImageGrid();
        } finally {
            state.isRunning = false;
            elements.runBtn.disabled = false;
            renderModelInfo();
        }
    };

    elements.modelDropzone.addEventListener("click", () => elements.modelInput.click());
    elements.modelDropzone.addEventListener("dragover", (evt) => {
        evt.preventDefault();
        elements.modelDropzone.classList.add("is-dragover");
    });
    elements.modelDropzone.addEventListener("dragleave", () =>
        elements.modelDropzone.classList.remove("is-dragover"),
    );
    elements.modelDropzone.addEventListener("drop", (evt) => {
        evt.preventDefault();
        elements.modelDropzone.classList.remove("is-dragover");
        handleModelFiles(evt.dataTransfer.files);
    });
    elements.modelInput.addEventListener("change", (evt) => {
        handleModelFiles(evt.target.files);
        elements.modelInput.value = "";
    });

    elements.imageDropzone.addEventListener("click", () => elements.imageInput.click());
    elements.imageDropzone.addEventListener("dragover", (evt) => {
        evt.preventDefault();
        elements.imageDropzone.classList.add("is-dragover");
    });
    elements.imageDropzone.addEventListener("dragleave", () =>
        elements.imageDropzone.classList.remove("is-dragover"),
    );
    elements.imageDropzone.addEventListener("drop", (evt) => {
        evt.preventDefault();
        elements.imageDropzone.classList.remove("is-dragover");
        handleImageFiles(evt.dataTransfer.files);
    });
    elements.imageInput.addEventListener("change", (evt) => {
        handleImageFiles(evt.target.files);
        elements.imageInput.value = "";
    });

    elements.settingsForm.addEventListener("input", updateSettings);
    elements.runBtn.addEventListener("click", executeInference);
    elements.resetBtn.addEventListener("click", resetWorkspace);
    elements.classEditor.addEventListener("input", (evt) => {
        state.classEditorText = evt.target.value;
        renderClassPreview();
    });

    restorePrefilledClasses();
    elements.classEditor.value = state.classEditorText;
    renderModelInfo();
    renderImageGrid();
    renderLogs();
    renderClassPreview();
})();
