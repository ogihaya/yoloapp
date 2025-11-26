(() => {
    const script = document.currentScript;
    const config = {
        pageType: script?.dataset.pageType || "train",
        exportUrl: script?.dataset.exportUrl || "",
        exportName: script?.dataset.exportName || "annotations.zip",
        nextUrl: script?.dataset.nextUrl || "",
        inferenceUrl: script?.dataset.inferenceUrl || "",
        colabUrl: script?.dataset.colabUrl || "",
    };
    const isTrain = config.pageType === "train";
    const isVal = config.pageType === "val";

    const STORAGE_KEYS = {
        valPrefill: "yoloApp.prefillValClasses",
        inferencePrefill: "yoloApp.prefillInferenceClasses",
    };

    const CLASS_COLOR_PALETTE = [
        "#38bdf8",
        "#f472b6",
        "#facc15",
        "#34d399",
        "#a78bfa",
        "#fb7185",
        "#f97316",
        "#4ade80",
        "#c084fc",
    ];

    const state = {
        classes: [],
        activeClassId: null,
        images: [],
        viewSize: "medium",
        selectedImageIds: [],
        selectedBox: null,
    };

    const classListEl = document.getElementById("classList");
    const classForm = document.getElementById("classForm");
    const colorInput = document.getElementById("classColor");
    const classNameInput = document.getElementById("className");
    const classFormSubmit = document.getElementById("classFormSubmit");
    const cancelEditBtn = document.getElementById("cancelEditBtn");
    const activeClassDisplay = document.getElementById("activeClassDisplay");
    const dropzone = document.getElementById("dropzone");
    const imageInput = document.getElementById("imageInput");
    const imageGrid = document.getElementById("imageGrid");
    const viewSizeButtons = document.querySelectorAll("[data-view-size]");
    const toastEl = document.getElementById("toast");
    const exportModal = document.getElementById("exportModal");
    const exportBtn = document.getElementById("exportBtn");
    const confirmExport = document.getElementById("confirmExport");
    const nextStepBtn = document.getElementById("nextStepBtn");
    const colabBtn = document.getElementById("colabBtn");
    const goInferenceBtn = document.getElementById("goInferenceBtn");
    const panel = document.getElementById("classPanel");
    const panelToggle = document.querySelector("[data-panel-toggle]");
    const panelClose = document.querySelector("[data-panel-close]");
    const modalClose = document.querySelector("[data-modal-close]");
    const selectAllBtn = document.getElementById("selectAllImages");
    const clearSelectionBtn = document.getElementById("clearSelection");
    const deleteSelectedBtn = document.getElementById("deleteSelectedImages");
    const selectionCounter = document.getElementById("selectionCounter");

    let toastTimer = null;
    let drawingContext = null;
    let nextColorIndex = 0;
    let editingClassId = null;

    const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : `id-${Date.now()}-${Math.random()}`);

    const getClassById = (id) => state.classes.find((cls) => cls.id === id);

    const getSuggestedColor = () => CLASS_COLOR_PALETTE[nextColorIndex % CLASS_COLOR_PALETTE.length];
    const advanceColorSuggestion = () => {
        nextColorIndex = (nextColorIndex + 1) % CLASS_COLOR_PALETTE.length;
    };

    const refreshClassUI = () => {
        renderClassList();
        updateActiveClassDisplay();
    };

    const resetClassFormState = () => {
        editingClassId = null;
        classForm.reset();
        classNameInput.value = "";
        colorInput.value = getSuggestedColor();
        classFormSubmit.textContent = "クラスを追加";
        cancelEditBtn.hidden = true;
    };

    const enterEditMode = (cls) => {
        editingClassId = cls.id;
        classNameInput.value = cls.label;
        colorInput.value = cls.color;
        classFormSubmit.textContent = "クラスを更新";
        cancelEditBtn.hidden = false;
        state.activeClassId = cls.id;
        renderClassList();
        updateActiveClassDisplay();
    };

    const applyViewSize = () => {
        imageGrid.dataset.view = state.viewSize;
        viewSizeButtons.forEach((button) => {
            button.classList.toggle("is-active", button.dataset.viewSize === state.viewSize);
        });
    };

    const buildClassTransferPayload = () => ({
        classes: state.classes.map((cls) => ({ id: cls.id, label: cls.label, color: cls.color })),
        activeClassId: state.activeClassId,
        timestamp: Date.now(),
    });

    const restoreTransferredClasses = () => {
        const raw = sessionStorage.getItem(STORAGE_KEYS.valPrefill);
        if (!raw) return;
        sessionStorage.removeItem(STORAGE_KEYS.valPrefill);
        try {
            const payload = JSON.parse(raw);
            const incoming = Array.isArray(payload?.classes) ? payload.classes : [];
            if (!incoming.length) return;
            state.classes = incoming.map((cls, index) => ({
                id: cls.id || uuid(),
                label: cls.label || `Class ${index + 1}`,
                color: cls.color || "#38bdf8",
            }));
            const fallbackId = state.classes[0]?.id ?? null;
            const desiredActive = payload?.activeClassId;
            state.activeClassId = state.classes.some((cls) => cls.id === desiredActive) ? desiredActive : fallbackId;
            nextColorIndex = state.classes.length % CLASS_COLOR_PALETTE.length;
            resetClassFormState();
            renderClassList();
            updateActiveClassDisplay();
            showToast("Train画面のクラス設定を引き継ぎました");
        } catch (error) {
            console.error("Failed to restore classes from train page", error);
        }
    };

    const showToast = (message) => {
        toastEl.textContent = message;
        toastEl.classList.add("is-visible");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => toastEl.classList.remove("is-visible"), 2500);
    };

    const removeClass = (id) => {
        const target = getClassById(id);
        if (!target) return;
        state.classes = state.classes.filter((cls) => cls.id !== id);
        state.images.forEach((image) => {
            image.boxes = image.boxes.filter((box) => box.classId !== id);
        });
        if (state.activeClassId === id) {
            state.activeClassId = state.classes[0]?.id ?? null;
        }
        if (editingClassId === id) {
            resetClassFormState();
        }
        refreshClassUI();
        renderImages();
        showToast(`クラス「${target.label}」を削除しました`);
    };

    const removeImage = (imageId) => {
        const target = state.images.find((img) => img.id === imageId);
        if (!target) return;
        state.images = state.images.filter((img) => img.id !== imageId);
        state.selectedImageIds = state.selectedImageIds.filter((id) => id !== imageId);
        if (state.selectedBox?.imageId === imageId) {
            state.selectedBox = null;
        }
        renderImages();
        showToast(`画像「${target.name}」を削除しました`);
    };

    const renderClassList = () => {
        classListEl.innerHTML = "";
        if (!state.classes.length) {
            const empty = document.createElement("li");
            empty.className = "class-empty";
            empty.textContent = "まずはクラスを追加してください。";
            classListEl.appendChild(empty);
            return;
        }

        state.classes.forEach((cls) => {
            const li = document.createElement("li");
            li.className = "class-item";
            if (cls.id === state.activeClassId) li.classList.add("is-active");
            li.dataset.classId = cls.id;

            const tag = document.createElement("span");
            tag.className = "class-tag";

            const dot = document.createElement("span");
            dot.className = "class-dot";
            dot.style.background = cls.color;
            tag.appendChild(dot);

            const label = document.createElement("span");
            label.textContent = cls.label;
            tag.appendChild(label);

            li.appendChild(tag);
            const actions = document.createElement("div");
            actions.className = "class-item-actions";
            const count = document.createElement("small");
            const usage = state.images.reduce(
                (acc, img) => acc + img.boxes.filter((box) => box.classId === cls.id).length,
                0,
            );
            count.textContent = `${usage}個`;
            count.style.color = "var(--muted)";
            actions.appendChild(count);

            const editBtn = document.createElement("button");
            editBtn.type = "button";
            editBtn.textContent = "編集";
            editBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                enterEditMode(cls);
            });
            actions.appendChild(editBtn);

            if (cls.id === editingClassId) {
                const deleteBtn = document.createElement("button");
                deleteBtn.type = "button";
                deleteBtn.textContent = "削除";
                deleteBtn.addEventListener("click", (event) => {
                    event.stopPropagation();
                    if (!window.confirm(`クラス「${cls.label}」を削除しますか？`)) {
                        return;
                    }
                    removeClass(cls.id);
                });
                actions.appendChild(deleteBtn);
            }

            li.appendChild(actions);

            li.addEventListener("click", () => {
                state.activeClassId = cls.id;
                refreshClassUI();
            });

            classListEl.appendChild(li);
        });
    };

    const updateActiveClassDisplay = () => {
        const activeCls = getClassById(state.activeClassId);
        activeClassDisplay.textContent = activeCls ? activeCls.label : "—（未選択）";
    };

    const renderImages = () => {
        imageGrid.dataset.view = state.viewSize;
        imageGrid.innerHTML = "";
        state.selectedImageIds = state.selectedImageIds.filter((id) =>
            state.images.some((img) => img.id === id),
        );
        if (!state.images.length) {
            const empty = document.createElement("div");
            empty.className = "empty-state";
            empty.textContent = "追加した画像がここに表示されます。";
            imageGrid.appendChild(empty);
            updateSelectionControls();
            return;
        }

        state.images.forEach((image) => {
            const card = document.createElement("article");
            card.className = "image-card";
            const isSelected = state.selectedImageIds.includes(image.id);
            if (isSelected) card.classList.add("is-selected");
            card.dataset.imageId = image.id;

            const meta = document.createElement("div");
            meta.className = "image-meta";
            const metaInfo = document.createElement("div");
            metaInfo.className = "image-meta-info";
            metaInfo.innerHTML = `<span>${image.name}</span><span>${image.boxes.length} annotations</span>`;
            const selector = document.createElement("label");
            selector.className = "image-select";
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = isSelected;
            checkbox.addEventListener("change", (evt) => toggleImageSelection(image.id, evt.target.checked));
            selector.appendChild(checkbox);
            const selectText = document.createElement("span");
            selector.appendChild(selectText);
            meta.appendChild(selector);
            meta.appendChild(metaInfo);
            card.appendChild(meta);

            const stage = document.createElement("div");
            stage.className = "image-stage";
            const imgEl = document.createElement("img");
            imgEl.src = image.src;
            imgEl.alt = image.name;
            stage.appendChild(imgEl);

            const layer = document.createElement("div");
            layer.className = "annotation-layer";
            layer.dataset.imageId = image.id;
            layer.addEventListener("pointerdown", startDrawing);
            layer.addEventListener("pointermove", resizeDrawing);
            layer.addEventListener("pointerup", finishDrawing);
            layer.addEventListener("pointerleave", cancelDrawing);
            stage.appendChild(layer);

            renderBoxes(layer, image);

            card.appendChild(stage);

            const list = document.createElement("ul");
            list.className = "box-list";
            if (!image.boxes.length) {
                const emptyMsg = document.createElement("p");
                emptyMsg.className = "muted";
                emptyMsg.textContent = "矩形を追加するとここに一覧表示されます。";
                card.appendChild(emptyMsg);
            } else {
                image.boxes.forEach((box) => {
                    const li = document.createElement("li");
                    const cls = getClassById(box.classId);
                    li.textContent = `${cls ? cls.label : "Unknown"} (${box.w.toFixed(1)}% × ${box.h.toFixed(1)}%)`;
                    list.appendChild(li);
                });
                card.appendChild(list);
            }

            const actions = document.createElement("div");
            actions.className = "image-actions";
            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.className = "btn-outline btn-danger";
            deleteBtn.textContent = "画像を削除";
            deleteBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                if (window.confirm(`画像「${image.name}」を削除しますか？`)) {
                    removeImage(image.id);
                }
            });
            actions.appendChild(deleteBtn);
            card.appendChild(actions);

            imageGrid.appendChild(card);
        });

        updateSelectionControls();
    };

    const renderBoxes = (layer, image) => {
        layer.querySelectorAll(".annotation-box").forEach((boxEl) => boxEl.remove());
        image.boxes.forEach((box) => {
            const cls = getClassById(box.classId);
            const el = document.createElement("div");
            el.className = "annotation-box";
            el.dataset.boxId = box.id;
            el.dataset.imageId = image.id;
            el.style.left = `${box.x}%`;
            el.style.top = `${box.y}%`;
            el.style.width = `${box.w}%`;
            el.style.height = `${box.h}%`;
            el.style.borderColor = box.color;
            if (state.selectedBox?.boxId === box.id && state.selectedBox?.imageId === image.id) {
                el.classList.add("is-selected");
            }

            const label = document.createElement("span");
            label.style.background = `${box.color}cc`;
            label.textContent = cls ? cls.label : "Class";
            el.appendChild(label);

            const btn = document.createElement("button");
            btn.type = "button";
            btn.ariaLabel = "削除";
            btn.textContent = "×";
            btn.hidden = !(state.selectedBox?.boxId === box.id && state.selectedBox?.imageId === image.id);
            btn.addEventListener("pointerdown", (event) => {
                // Prevent starting a new drawing when clicking the delete button
                event.stopPropagation();
            });
            btn.addEventListener("click", (event) => {
                event.stopPropagation();
                image.boxes = image.boxes.filter((candidate) => candidate.id !== box.id);
                if (state.selectedBox?.boxId === box.id && state.selectedBox?.imageId === image.id) {
                    state.selectedBox = null;
                }
                renderImages();
            });
            el.appendChild(btn);

            el.addEventListener("pointerdown", (event) => {
                event.stopPropagation();
                setSelectedBox(image.id, box.id);
            });

            layer.appendChild(el);
        });
    };

    const pointerPosition = (evt, element) => {
        const rect = element.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top,
            width: rect.width,
            height: rect.height,
        };
    };

    const startDrawing = (evt) => {
        // Ignore interactions that originate from existing annotations (e.g., delete buttons)
        if (evt.target !== evt.currentTarget) {
            return;
        }
        clearSelectedBox();
        if (!state.classes.length) {
            showToast("クラスを追加してからアノテーションしてください");
            return;
        }
        if (!state.activeClassId) {
            showToast("クラスを先に選択してください");
            return;
        }
        const layer = evt.currentTarget;
        const image = state.images.find((img) => img.id === layer.dataset.imageId);
        if (!image) return;
        layer.setPointerCapture(evt.pointerId);
        const pos = pointerPosition(evt, layer);
        const cls = getClassById(state.activeClassId);
        const ghost = document.createElement("div");
        ghost.className = "annotation-box is-drawing";
        ghost.style.borderColor = cls?.color || "var(--accent)";
        layer.appendChild(ghost);
        drawingContext = {
            image,
            layer,
            startX: pos.x,
            startY: pos.y,
            width: pos.width,
            height: pos.height,
            ghost,
            pointerId: evt.pointerId,
        };
        evt.preventDefault();
    };

    const resizeDrawing = (evt) => {
        if (!drawingContext || evt.currentTarget !== drawingContext.layer) return;
        const pos = pointerPosition(evt, drawingContext.layer);
        updateGhostBox(pos);
    };

    const finishDrawing = (evt) => {
        if (!drawingContext || evt.currentTarget !== drawingContext.layer) return;
        const pos = pointerPosition(evt, drawingContext.layer);
        const { widthPercent, heightPercent } = updateGhostBox(pos);
        drawingContext.layer.releasePointerCapture(drawingContext.pointerId);
        drawingContext.ghost.remove();
        if (widthPercent < 2 || heightPercent < 2) {
            drawingContext = null;
            return;
        }
        const cls = getClassById(state.activeClassId);
        drawingContext.image.boxes.push({
            id: uuid(),
            classId: state.activeClassId,
            color: cls?.color ?? "var(--accent)",
            x: drawingContext.leftPercent,
            y: drawingContext.topPercent,
            w: widthPercent,
            h: heightPercent,
        });
        drawingContext = null;
        renderImages();
    };

    const cancelDrawing = (evt) => {
        if (!drawingContext || evt.currentTarget !== drawingContext.layer) return;
        drawingContext.ghost.remove();
        drawingContext.layer.releasePointerCapture(drawingContext.pointerId);
        drawingContext = null;
    };

    const updateGhostBox = (pos) => {
        if (!drawingContext) return { widthPercent: 0, heightPercent: 0 };
        const clamp = (value, max) => Math.max(0, Math.min(value, max));
        const x = clamp(pos.x, drawingContext.width);
        const y = clamp(pos.y, drawingContext.height);
        const left = Math.min(drawingContext.startX, x);
        const top = Math.min(drawingContext.startY, y);
        const width = Math.abs(x - drawingContext.startX);
        const height = Math.abs(y - drawingContext.startY);
        const widthPercent = (width / drawingContext.width) * 100;
        const heightPercent = (height / drawingContext.height) * 100;
        drawingContext.leftPercent = (left / drawingContext.width) * 100;
        drawingContext.topPercent = (top / drawingContext.height) * 100;
        drawingContext.ghost.style.left = `${drawingContext.leftPercent}%`;
        drawingContext.ghost.style.top = `${drawingContext.topPercent}%`;
        drawingContext.ghost.style.width = `${widthPercent}%`;
        drawingContext.ghost.style.height = `${heightPercent}%`;
        return { widthPercent, heightPercent };
    };

    const handleFiles = (files) => {
        Array.from(files).forEach((file) => {
            if (!file.type.startsWith("image/")) return;
            const reader = new FileReader();
            reader.onload = (event) => {
                state.images.push({
                    id: uuid(),
                    name: file.name,
                    src: event.target.result,
                    boxes: [],
                });
                renderImages();
            };
            reader.readAsDataURL(file);
        });
        showToast(`${files.length}件のファイルを読み込みました`);
    };

    const toggleImageSelection = (imageId, selected) => {
        if (selected) {
            if (!state.selectedImageIds.includes(imageId)) {
                state.selectedImageIds.push(imageId);
            }
        } else {
            state.selectedImageIds = state.selectedImageIds.filter((id) => id !== imageId);
        }
        updateSelectionControls();
        refreshBoxSelection();
    };

    const selectAllImages = () => {
        state.selectedImageIds = state.images.map((img) => img.id);
        updateSelectionControls();
        renderImages();
    };

    const clearImageSelection = () => {
        if (!state.selectedImageIds.length) return;
        state.selectedImageIds = [];
        updateSelectionControls();
        renderImages();
    };

    const deleteSelectedImages = () => {
        if (!state.selectedImageIds.length) {
            showToast("削除する画像を選択してください");
            return;
        }
        if (!window.confirm(`選択した${state.selectedImageIds.length}枚の画像を削除しますか？`)) {
            return;
        }
        const names = state.images
            .filter((img) => state.selectedImageIds.includes(img.id))
            .map((img) => img.name);
        state.images = state.images.filter((img) => !state.selectedImageIds.includes(img.id));
        state.selectedImageIds = [];
        state.selectedBox = null;
        renderImages();
        showToast(`${names.length}枚の画像を削除しました`);
    };

    const updateSelectionControls = () => {
        if (selectionCounter) {
            selectionCounter.hidden = !state.selectedImageIds.length;
            selectionCounter.textContent = `${state.selectedImageIds.length}枚選択中`;
        }
        if (deleteSelectedBtn) {
            deleteSelectedBtn.disabled = !state.selectedImageIds.length;
        }
        if (clearSelectionBtn) {
            clearSelectionBtn.disabled = !state.selectedImageIds.length;
        }
        if (selectAllBtn) {
            selectAllBtn.disabled = !state.images.length || state.selectedImageIds.length === state.images.length;
        }
        refreshImageSelectionUI();
    };

    const setSelectedBox = (imageId, boxId) => {
        state.selectedBox = { imageId, boxId };
        refreshBoxSelection();
    };

    const clearSelectedBox = () => {
        if (!state.selectedBox) return;
        state.selectedBox = null;
        refreshBoxSelection();
    };

    const refreshBoxSelection = () => {
        document.querySelectorAll(".annotation-box").forEach((boxEl) => {
            const isSelected =
                state.selectedBox?.boxId === boxEl.dataset.boxId &&
                state.selectedBox?.imageId === boxEl.dataset.imageId;
            boxEl.classList.toggle("is-selected", Boolean(isSelected));
            const deleteBtn = boxEl.querySelector("button");
            if (deleteBtn) deleteBtn.hidden = !isSelected;
        });
    };

    const refreshImageSelectionUI = () => {
        const selectedSet = new Set(state.selectedImageIds);
        document.querySelectorAll(".image-card").forEach((card) => {
            const imageId = card.dataset.imageId;
            const isSelected = selectedSet.has(imageId);
            card.classList.toggle("is-selected", isSelected);
            const checkbox = card.querySelector('.image-select input[type="checkbox"]');
            if (checkbox) checkbox.checked = isSelected;
        });
    };

    dropzone.addEventListener("click", () => imageInput.click());
    dropzone.addEventListener("dragover", (evt) => {
        evt.preventDefault();
        dropzone.classList.add("is-dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragover"));
    dropzone.addEventListener("drop", (evt) => {
        evt.preventDefault();
        dropzone.classList.remove("is-dragover");
        if (evt.dataTransfer?.files?.length) {
            handleFiles(evt.dataTransfer.files);
        }
    });
    imageInput.addEventListener("change", (evt) => {
        if (evt.target.files?.length) {
            handleFiles(evt.target.files);
            imageInput.value = "";
        }
    });

    viewSizeButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const desired = button.dataset.viewSize;
            if (!desired || desired === state.viewSize) return;
            state.viewSize = desired;
            applyViewSize();
        });
    });

    classForm.addEventListener("submit", (evt) => {
        evt.preventDefault();
        const formData = new FormData(classForm);
        const label = formData.get("className").toString().trim();
        if (!label) {
            showToast("クラス名を入力してください");
            return;
        }
        const color = formData.get("classColor")?.toString() || getSuggestedColor();
        const labelLower = label.toLowerCase();
        const hasDuplicate = state.classes.some(
            (cls) => cls.label.toLowerCase() === labelLower && cls.id !== editingClassId,
        );
        if (hasDuplicate) {
            showToast("同じ名前のクラスは追加できません");
            return;
        }

        if (editingClassId) {
            const target = getClassById(editingClassId);
            if (!target) {
                resetClassFormState();
                return;
            }
            target.label = label;
            target.color = color;
            state.images.forEach((image) => {
                image.boxes.forEach((box) => {
                    if (box.classId === target.id) {
                        box.color = color;
                    }
                });
            });
            refreshClassUI();
            renderImages();
            showToast(`クラス「${label}」を更新しました`);
            resetClassFormState();
            return;
        }

        const newClass = { id: uuid(), label, color };
        state.classes.push(newClass);
        state.activeClassId = newClass.id;
        refreshClassUI();
        showToast(`クラス「${label}」を追加しました`);
        advanceColorSuggestion();
        resetClassFormState();
    });

    cancelEditBtn.addEventListener("click", () => {
        resetClassFormState();
        refreshClassUI();
    });

    if (selectAllBtn) {
        selectAllBtn.addEventListener("click", selectAllImages);
    }
    if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener("click", clearImageSelection);
    }
    if (deleteSelectedBtn) {
        deleteSelectedBtn.addEventListener("click", deleteSelectedImages);
    }

    exportBtn.addEventListener("click", () => {
        if (!state.images.length) {
            showToast("まずは画像をインポートしてください");
            return;
        }
        if (!state.classes.length) {
            showToast("エクスポートするクラスを作成してください");
            return;
        }
        exportModal.setAttribute("aria-hidden", "false");
    });
    modalClose.addEventListener("click", () => exportModal.setAttribute("aria-hidden", "true"));
    confirmExport.addEventListener("click", async () => {
        exportModal.setAttribute("aria-hidden", "true");
        await performExport();
    });

    if (nextStepBtn && config.nextUrl) {
        nextStepBtn.addEventListener("click", () => {
            if (!state.classes.length) {
                showToast("次のステップに進む前にクラスを追加してください");
                return;
            }
            sessionStorage.setItem(STORAGE_KEYS.valPrefill, JSON.stringify(buildClassTransferPayload()));
            window.location.href = config.nextUrl;
        });
    }

    if (colabBtn && config.colabUrl) {
        colabBtn.addEventListener("click", () => {
            const win = window.open(config.colabUrl, "_blank", "noopener,noreferrer");
            if (win) {
                win.opener = null;
            }
        });
    }

    if (goInferenceBtn && config.inferenceUrl) {
        goInferenceBtn.addEventListener("click", () => {
            if (!state.classes.length) {
                showToast("推論画面へ移動する前にクラスを設定してください");
                return;
            }
            sessionStorage.setItem(STORAGE_KEYS.inferencePrefill, JSON.stringify(buildClassTransferPayload()));
            window.location.href = config.inferenceUrl;
        });
    }

    const performExport = async () => {
        if (!config.exportUrl) {
            showToast("エクスポート先が設定されていません");
            return;
        }
        confirmExport.disabled = true;
        confirmExport.textContent = "エクスポート中...";
        try {
            const response = await fetch(config.exportUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    classes: state.classes,
                    images: state.images,
                }),
            });

            if (!response.ok) {
                let message = "エクスポートに失敗しました";
                try {
                    const data = await response.json();
                    if (data?.error) message = data.error;
                } catch (_) {
                    // ignore JSON parse errors
                }
                throw new Error(message);
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = config.exportName || "annotations.zip";
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => window.URL.revokeObjectURL(url), 2000);
            showToast(`${config.exportName || "エクスポート"}をダウンロードしました`);
        } catch (error) {
            console.error(error);
            showToast(error.message || "エクスポートに失敗しました");
        } finally {
            confirmExport.disabled = false;
            confirmExport.textContent = "エクスポート";
        }
    };

    panelToggle.addEventListener("click", () => panel.classList.toggle("is-visible"));
    if (panelClose) {
        panelClose.addEventListener("click", () => panel.classList.remove("is-visible"));
    }

    resetClassFormState();
    applyViewSize();
    renderClassList();
    updateActiveClassDisplay();
    updateSelectionControls();
    if (isVal) {
        restoreTransferredClasses();
    }
})();
