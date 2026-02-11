(function () {
    document.documentElement.classList.add("js-enabled");
    const PAGE_CLASS_PREFIX = "page-";
    const KNOWN_PAGES = ["home", "explore", "themes", "extract", "settings"];

    function byId(id) {
        return document.getElementById(id);
    }

    function getPageFromPath(pathname) {
        if (pathname.startsWith("/explore-pfds")) {
            return "explore";
        }
        if (pathname.startsWith("/filter")) {
            return "explore";
        }
        if (pathname.startsWith("/analyse-themes")) {
            return "themes";
        }
        if (pathname.startsWith("/extract-data")) {
            return "extract";
        }
        if (pathname.startsWith("/settings")) {
            return "settings";
        }
        return "home";
    }

    function applyBodyPageClass(page) {
        KNOWN_PAGES.forEach((name) => {
            document.body.classList.remove(`${PAGE_CLASS_PREFIX}${name}`);
        });
        document.body.classList.add(`${PAGE_CLASS_PREFIX}${page}`);
    }

    function syncSidebarActive(page) {
        const links = document.querySelectorAll(".sidebar-nav [data-page-link]");
        links.forEach((link) => {
            const isActive = link.getAttribute("data-page-link") === page;
            link.classList.toggle("is-active", isActive);
            if (isActive) {
                link.setAttribute("aria-current", "page");
            } else {
                link.removeAttribute("aria-current");
            }
        });
    }

    function toggleProviderFields() {
        const provider = byId("provider_override");
        const openai = byId("openai-fields");
        const openrouter = byId("openrouter-fields");
        if (!provider || !openai || !openrouter) {
            return;
        }
        const isOpenRouter = provider.value === "OpenRouter";
        openai.classList.toggle("hidden", isOpenRouter);
        openrouter.classList.toggle("hidden", !isOpenRouter);
    }

    function toggleDiscoverTrimFields() {
        const trimApproach = byId("trim_approach");
        const truncationBlock = byId("discover-truncation-block");
        const summaryBlock = byId("discover-summary-block");
        if (!trimApproach || !truncationBlock || !summaryBlock) {
            return;
        }
        const isTruncate = trimApproach.value === "truncate";
        truncationBlock.classList.toggle("hidden", !isTruncate);
        summaryBlock.classList.toggle("hidden", isTruncate);
    }

    function toggleTruncationTypeFields() {
        const limitType = byId("truncation_limit_type");
        const tokensBlock = byId("discover-tokens-block");
        const wordsBlock = byId("discover-words-block");
        if (!limitType || !tokensBlock || !wordsBlock) {
            return;
        }
        const showWords = limitType.value === "words";
        tokensBlock.classList.toggle("hidden", showWords);
        wordsBlock.classList.toggle("hidden", !showWords);
    }

    function setupFeatureGrid() {
        const tableBody = byId("feature-grid-body");
        const addButton = byId("add-feature-row");
        if (!tableBody || !addButton) {
            return;
        }

        function bindRemoveButtons() {
            const removeButtons = tableBody.querySelectorAll(".remove-feature-row");
            removeButtons.forEach((btn) => {
                btn.onclick = function () {
                    const row = btn.closest("tr");
                    if (!row) {
                        return;
                    }
                    const rows = tableBody.querySelectorAll("tr");
                    if (rows.length <= 1) {
                        row.querySelectorAll("input").forEach((input) => {
                            input.value = "";
                        });
                        const select = row.querySelector("select");
                        if (select) {
                            select.value = "Free text";
                        }
                        return;
                    }
                    row.remove();
                };
            });
        }

        addButton.onclick = function () {
            const firstRow = tableBody.querySelector("tr");
            if (!firstRow) {
                return;
            }
            const clone = firstRow.cloneNode(true);
            clone.querySelectorAll("input").forEach((input) => {
                input.value = "";
            });
            const select = clone.querySelector("select");
            if (select) {
                select.value = "Free text";
            }
            tableBody.appendChild(clone);
            bindRemoveButtons();
        };

        bindRemoveButtons();
    }

    function setupTableScrollbars() {
        const osGlobal = window.OverlayScrollbarsGlobal;
        if (!osGlobal || typeof osGlobal.OverlayScrollbars !== "function") {
            return;
        }

        const OverlayScrollbars = osGlobal.OverlayScrollbars;
        const tableWraps = document.querySelectorAll(".explore-surface .table-wrap");
        tableWraps.forEach((tableWrap) => {
            if (OverlayScrollbars.valid && OverlayScrollbars.valid(tableWrap)) {
                return;
            }

            OverlayScrollbars(tableWrap, {
                overflow: {
                    x: "scroll",
                    y: "scroll",
                },
                scrollbars: {
                    theme: "os-theme-workbench",
                    autoHide: "move",
                    autoHideDelay: 520,
                    clickScroll: true,
                },
            });
        });
    }

    function setupPageScrollbar() {
        const osGlobal = window.OverlayScrollbarsGlobal;
        if (!osGlobal || typeof osGlobal.OverlayScrollbars !== "function") {
            return;
        }

        const OverlayScrollbars = osGlobal.OverlayScrollbars;
        const pageRoot = document.body;
        if (!pageRoot) {
            return;
        }
        if (OverlayScrollbars.valid && OverlayScrollbars.valid(pageRoot)) {
            return;
        }

        OverlayScrollbars(pageRoot, {
            scrollbars: {
                theme: "os-theme-workbench",
                autoHide: "move",
                autoHideDelay: 520,
                clickScroll: true,
            },
        });
    }

    function setupDatasetCellPreview() {
        const datasetTable = document.querySelector(".data-card .data-table");
        if (!datasetTable) {
            return;
        }
        if (datasetTable.dataset.previewBound === "1") {
            return;
        }
        datasetTable.dataset.previewBound = "1";

        const headerCells = Array.from(datasetTable.querySelectorAll("thead th"));
        const bodyRows = Array.from(datasetTable.querySelectorAll("tbody tr"));
        if (bodyRows.length > 120) {
            return;
        }
        if (!headerCells.length) {
            return;
        }

        function getColumnClass(headerText) {
            const label = (headerText || "").trim().toLowerCase();
            if (label === "id") {
                return "dataset-col-id";
            }
            if (label.includes("date")) {
                return "dataset-col-date";
            }
            if (label.includes("url")) {
                return "dataset-col-url";
            }
            if (
                label.includes("investigation") ||
                label.includes("circumstances") ||
                label.includes("concern")
            ) {
                return "dataset-col-long";
            }
            if (
                label.includes("coroner") ||
                label.includes("area") ||
                label.includes("receiver")
            ) {
                return "dataset-col-meta";
            }
            return "dataset-col-standard";
        }

        const columnClasses = headerCells.map((th) => {
            const columnClass = getColumnClass(th.innerText);
            th.classList.add(columnClass);
            return columnClass;
        });

        bodyRows.forEach((row) => {
            Array.from(row.children).forEach((cell, index) => {
                const columnClass = columnClasses[index];
                if (columnClass) {
                    cell.classList.add(columnClass);
                }
            });
        });

        const cells = datasetTable.querySelectorAll("tbody td");
        if (!cells.length) {
            return;
        }

        const popup = document.createElement("div");
        popup.className = "dataset-cell-popup";
        popup.setAttribute("role", "dialog");
        popup.setAttribute("aria-modal", "false");
        popup.setAttribute("aria-label", "Full cell text");

        const popupContent = document.createElement("div");
        popupContent.className = "dataset-cell-popup__content";
        popup.appendChild(popupContent);
        document.body.appendChild(popup);

        const textMeasureCanvas = document.createElement("canvas");
        const textMeasureContext = textMeasureCanvas.getContext("2d");
        let activeCell = null;

        function decodeEscapedNewlines(value) {
            return String(value || "")
                .replace(/\\r\\n/g, "\n")
                .replace(/\\n/g, "\n")
                .replace(/\\r/g, "\n");
        }

        function clampNumber(value, min, max) {
            return Math.min(max, Math.max(min, value));
        }

        function measureLongestLinePx(text) {
            if (!textMeasureContext) {
                return 0;
            }
            const font = window.getComputedStyle(popupContent).font;
            if (font) {
                textMeasureContext.font = font;
            }
            const lines = String(text || "").split("\n");
            let widest = 0;
            lines.forEach((line) => {
                const width = textMeasureContext.measureText(line).width;
                widest = Math.max(widest, width);
            });
            return widest;
        }

        function computePopupWidthPx(text, cell) {
            const totalLength = String(text || "").trim().length;
            const isVeryShortField =
                cell.classList.contains("dataset-col-id") ||
                cell.classList.contains("dataset-col-date") ||
                cell.classList.contains("dataset-col-meta");
            const textPx = measureLongestLinePx(text);
            const contentPaddingPx = 34;

            if (isVeryShortField) {
                if (totalLength <= 28) {
                    return clampNumber(textPx + contentPaddingPx, 95, 220);
                }
                const compactTargetPx = Math.max(textPx + contentPaddingPx, Math.sqrt(totalLength) * 16);
                return clampNumber(compactTargetPx, 120, 280);
            }

            if (totalLength <= 24) {
                return clampNumber(textPx + contentPaddingPx, 120, 340);
            }

            const targetPx = Math.max(textPx + contentPaddingPx, Math.sqrt(totalLength) * 22);
            return clampNumber(targetPx, 220, 700);
        }

        function hidePopup() {
            popup.classList.remove("is-visible");
            popup.style.width = "";
            if (activeCell) {
                activeCell.classList.remove("is-active");
                activeCell = null;
            }
        }

        function positionPopup(cell) {
            const margin = 12;
            const cellRect = cell.getBoundingClientRect();
            const popupRect = popup.getBoundingClientRect();

            let left = cellRect.left;
            let top = cellRect.bottom + 8;

            if (left + popupRect.width > window.innerWidth - margin) {
                left = window.innerWidth - popupRect.width - margin;
            }
            if (left < margin) {
                left = margin;
            }
            if (top + popupRect.height > window.innerHeight - margin) {
                top = cellRect.top - popupRect.height - 8;
            }
            if (top < margin) {
                top = margin;
            }

            popup.style.left = `${left}px`;
            popup.style.top = `${top}px`;
        }

        function showPopup(cell) {
            const fullText = (cell.dataset.fullText || "").trim();
            if (!fullText) {
                return;
            }

            const decodedText = decodeEscapedNewlines(fullText);
            if (activeCell && activeCell !== cell) {
                activeCell.classList.remove("is-active");
            }

            activeCell = cell;
            activeCell.classList.add("is-active");
            popup.style.width = `${Math.round(computePopupWidthPx(decodedText, cell))}px`;
            popupContent.textContent = decodedText;
            popup.classList.add("is-visible");
            positionPopup(cell);
        }

        cells.forEach((cell) => {
            const fullText = (cell.innerText || "").trim();
            if (!fullText) {
                return;
            }

            cell.dataset.fullText = fullText;
            const clamp = document.createElement("div");
            clamp.className = "dataset-cell-clamp";
            clamp.textContent = fullText;
            cell.textContent = "";
            cell.appendChild(clamp);
            cell.classList.add("dataset-cell-previewable");
            cell.setAttribute("tabindex", "0");
            cell.setAttribute("role", "button");
            cell.setAttribute("aria-label", "Show full cell text");

            cell.addEventListener("click", function (event) {
                event.stopPropagation();
                if (activeCell === cell && popup.classList.contains("is-visible")) {
                    hidePopup();
                    return;
                }
                showPopup(cell);
            });

            cell.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    showPopup(cell);
                }
                if (event.key === "Escape") {
                    hidePopup();
                }
            });
        });

        document.addEventListener("click", function (event) {
            if (!popup.classList.contains("is-visible")) {
                return;
            }
            if (popup.contains(event.target)) {
                return;
            }
            const clickedCell = event.target.closest(".data-card .data-table tbody td");
            if (clickedCell) {
                return;
            }
            hidePopup();
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                hidePopup();
            }
        });

        const datasetWrap = document.querySelector(".data-card .table-wrap");
        if (datasetWrap) {
            const viewport = datasetWrap.querySelector("[data-overlayscrollbars-viewport]");
            if (viewport) {
                viewport.addEventListener("scroll", hidePopup, { passive: true });
            }
            datasetWrap.addEventListener("scroll", hidePopup, { passive: true });
        }

        window.addEventListener("resize", hidePopup);
    }

    function setupSidebarCollapse() {
        const toggleButton = byId("sidebar-toggle");
        if (!toggleButton) {
            return;
        }
        if (toggleButton.dataset.bound === "1") {
            return;
        }
        toggleButton.dataset.bound = "1";

        const storageKey = "workbench.sidebarCollapsed";

        function setCollapsedState(collapsed) {
            document.body.classList.toggle("sidebar-collapsed", collapsed);
            toggleButton.setAttribute("aria-expanded", String(!collapsed));
            try {
                window.localStorage.setItem(storageKey, collapsed ? "1" : "0");
            } catch (error) {
                // Ignore storage issues.
            }
        }

        let collapsed = false;
        try {
            collapsed = window.localStorage.getItem(storageKey) === "1";
        } catch (error) {
            collapsed = false;
        }

        if (window.innerWidth <= 980) {
            collapsed = false;
        }
        setCollapsedState(collapsed);

        toggleButton.addEventListener("click", function () {
            const isCollapsed = document.body.classList.contains("sidebar-collapsed");
            setCollapsedState(!isCollapsed);
        });

        window.addEventListener("resize", function () {
            if (window.innerWidth <= 980 && document.body.classList.contains("sidebar-collapsed")) {
                setCollapsedState(false);
            }
        });
    }

    function setupRevealAnimations() {
        const revealNodes = Array.from(document.querySelectorAll(".reveal-up"));
        if (!revealNodes.length) {
            return;
        }

        if (!("IntersectionObserver" in window)) {
            revealNodes.forEach((node) => node.classList.add("is-visible"));
            return;
        }

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("is-visible");
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.15,
                rootMargin: "0px 0px -8% 0px",
            }
        );

        revealNodes.forEach((node, index) => {
            node.style.transitionDelay = `${Math.min(index * 45, 180)}ms`;
            observer.observe(node);
        });
    }

    function setupStartOverConfirm() {
        const openButton = document.querySelector("[data-start-over-open]");
        const cancelButton = document.querySelector("[data-start-over-cancel]");
        const modal = byId("start-over-confirm");
        if (!openButton || !cancelButton || !modal) {
            return;
        }
        if (openButton.dataset.bound === "1") {
            return;
        }
        openButton.dataset.bound = "1";

        function openModal() {
            modal.classList.remove("hidden");
            document.body.classList.add("modal-open");
            cancelButton.focus();
        }

        function closeModal() {
            modal.classList.add("hidden");
            document.body.classList.remove("modal-open");
            openButton.focus();
        }

        openButton.addEventListener("click", openModal);
        cancelButton.addEventListener("click", closeModal);

        modal.addEventListener("click", function (event) {
            if (event.target === modal) {
                closeModal();
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !modal.classList.contains("hidden")) {
                closeModal();
            }
        });
    }

    function setupDownloadBundleModal() {
        const openButton = document.querySelector("[data-download-open]");
        const cancelButton = document.querySelector("[data-download-cancel]");
        const popover = byId("download-popover");
        if (!openButton || !cancelButton || !popover) {
            return;
        }
        if (openButton.dataset.bound === "1") {
            return;
        }
        openButton.dataset.bound = "1";

        function openPopover() {
            popover.classList.remove("hidden");
        }

        function closePopover(restoreFocus) {
            popover.classList.add("hidden");
            if (restoreFocus) {
                openButton.focus();
            }
        }

        openButton.addEventListener("click", function () {
            if (popover.classList.contains("hidden")) {
                openPopover();
            } else {
                closePopover(false);
            }
        });

        cancelButton.addEventListener("click", function () {
            closePopover(true);
        });

        document.addEventListener("pointerdown", function (event) {
            if (popover.classList.contains("hidden")) {
                return;
            }
            if (popover.contains(event.target) || openButton.contains(event.target)) {
                return;
            }
            closePopover(false);
        }, true);

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !popover.classList.contains("hidden")) {
                closePopover(true);
            }
        });
    }

    function setupTopbarQuickSettings() {
        const openButtons = document.querySelectorAll("[data-topbar-open]");
        const closeButtons = document.querySelectorAll("[data-topbar-close]");
        const popovers = Array.from(document.querySelectorAll(".topbar-popover-backdrop"));
        if (!openButtons.length || !popovers.length) {
            return;
        }
        if (document.body.dataset.topbarQuickSettingsBound === "1") {
            return;
        }
        document.body.dataset.topbarQuickSettingsBound = "1";

        const popoverByKey = {
            "report-limit": byId("report-limit-popover"),
            "date-range": byId("date-range-popover"),
            "llm-settings": byId("llm-settings-popover"),
        };

        function closeAllPopovers(restoreFocusTarget) {
            popovers.forEach((popover) => {
                if (popover) {
                    popover.classList.add("hidden");
                }
            });
            if (restoreFocusTarget) {
                restoreFocusTarget.focus();
            }
        }

        openButtons.forEach((button) => {
            button.addEventListener("click", function () {
                const key = button.getAttribute("data-topbar-open");
                const targetPopover = popoverByKey[key];
                if (!targetPopover) {
                    return;
                }
                closeAllPopovers();
                targetPopover.classList.remove("hidden");
            });
        });

        closeButtons.forEach((button) => {
            button.addEventListener("click", function () {
                closeAllPopovers();
            });
        });

        popovers.forEach((popover) => {
            popover.addEventListener("click", function (event) {
                if (event.target === popover) {
                    closeAllPopovers();
                }
            });
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closeAllPopovers();
            }
        });
    }

    function setupTopbarLimitControls() {
        const slider = document.querySelector("[data-topbar-limit-slider]");
        const numberInput = document.querySelector("[data-topbar-limit-number]");
        if (!slider || !numberInput) {
            return;
        }
        if (slider.dataset.bound === "1") {
            return;
        }
        slider.dataset.bound = "1";

        function clampToSlider(value) {
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) {
                return Number(slider.min || 1);
            }
            const min = Number(slider.min || 1);
            const max = Number(slider.max || 5000);
            return Math.max(min, Math.min(max, Math.round(parsed)));
        }

        slider.addEventListener("input", function () {
            numberInput.value = String(clampToSlider(slider.value));
        });

        numberInput.addEventListener("input", function () {
            slider.value = String(clampToSlider(numberInput.value));
        });
    }

    function setupTopbarLLMProviderToggle() {
        const provider = byId("topbar_provider_override");
        const openaiFields = byId("topbar-openai-fields");
        const openrouterFields = byId("topbar-openrouter-fields");
        if (!provider || !openaiFields || !openrouterFields) {
            return;
        }
        if (provider.dataset.bound === "1") {
            return;
        }
        provider.dataset.bound = "1";

        function paintProviderFields() {
            const isOpenRouter = provider.value === "OpenRouter";
            openaiFields.classList.toggle("hidden", isOpenRouter);
            openrouterFields.classList.toggle("hidden", !isOpenRouter);
        }

        provider.addEventListener("change", paintProviderFields);
        paintProviderFields();
    }

    function setupTopbarAIResetConfirm() {
        const confirmModal = byId("topbar-ai-reset-confirm");
        if (!confirmModal) {
            return;
        }
        if (confirmModal.dataset.bound === "1") {
            return;
        }
        confirmModal.dataset.bound = "1";

        const forms = document.querySelectorAll("form.js-ai-reset-form");
        if (!forms.length) {
            return;
        }

        const cancelButton = confirmModal.querySelector("[data-topbar-ai-reset-cancel]");
        const confirmButton = confirmModal.querySelector("[data-topbar-ai-reset-confirm]");
        const aiFeaturesUsed = confirmModal.getAttribute("data-ai-features-used") === "1";
        let pendingForm = null;

        function closeConfirm() {
            confirmModal.classList.add("hidden");
            pendingForm = null;
        }

        function openConfirm(form) {
            pendingForm = form;
            confirmModal.classList.remove("hidden");
        }

        forms.forEach((form) => {
            form.addEventListener("submit", function (event) {
                if (!aiFeaturesUsed) {
                    return;
                }
                if (form.dataset.aiResetConfirmed === "1") {
                    delete form.dataset.aiResetConfirmed;
                    return;
                }
                event.preventDefault();
                openConfirm(form);
            });
        });

        if (cancelButton) {
            cancelButton.addEventListener("click", closeConfirm);
        }

        if (confirmButton) {
            confirmButton.addEventListener("click", function () {
                if (!pendingForm) {
                    closeConfirm();
                    return;
                }
                pendingForm.dataset.aiResetConfirmed = "1";
                const formToSubmit = pendingForm;
                closeConfirm();
                formToSubmit.requestSubmit();
            });
        }

        confirmModal.addEventListener("click", function (event) {
            if (event.target === confirmModal) {
                closeConfirm();
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !confirmModal.classList.contains("hidden")) {
                closeConfirm();
            }
        });
    }

    function setupTopbarDateValidation() {
        const form = byId("date-range-popover") ? byId("date-range-popover").querySelector("form") : null;
        const startInput = byId("topbar_report_start_date");
        const endInput = byId("topbar_report_end_date");
        if (!form || !startInput || !endInput) {
            return;
        }
        if (form.dataset.dateValidationBound === "1") {
            return;
        }
        form.dataset.dateValidationBound = "1";

        function isValidDDMMYYYY(value) {
            const raw = String(value || "").trim();
            const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(raw);
            if (!match) {
                return false;
            }
            const day = Number(match[1]);
            const month = Number(match[2]);
            const year = Number(match[3]);
            const parsed = new Date(year, month - 1, day);
            return (
                parsed.getFullYear() === year
                && parsed.getMonth() === month - 1
                && parsed.getDate() === day
            );
        }

        function validateField(input) {
            const value = String(input.value || "").trim();
            if (!value) {
                input.setCustomValidity("Enter a date in DD/MM/YYYY format.");
                return false;
            }
            if (!isValidDDMMYYYY(value)) {
                input.setCustomValidity("Use a real date in DD/MM/YYYY format (e.g. 11/02/2026).");
                return false;
            }
            input.setCustomValidity("");
            return true;
        }

        [startInput, endInput].forEach((input) => {
            input.addEventListener("input", function () {
                validateField(input);
            });
            input.addEventListener("blur", function () {
                validateField(input);
            });
        });

        form.addEventListener("submit", function (event) {
            const startValid = validateField(startInput);
            const endValid = validateField(endInput);
            if (!startValid || !endValid) {
                event.preventDefault();
                form.reportValidity();
            }
        });
    }

    function setupAdvancedAIFilterPopover() {
        const openButton = document.querySelector("[data-advanced-ai-filter-open]");
        const cancelButton = document.querySelector("[data-advanced-ai-filter-cancel]");
        const backdrop = byId("advanced-ai-popover-backdrop");
        const form = document.querySelector("#advanced-ai-popover-backdrop form");
        if (!openButton || !cancelButton || !backdrop || !form) {
            return;
        }
        if (openButton.dataset.bound === "1") {
            return;
        }
        openButton.dataset.bound = "1";

        const queryInput = byId("advanced_ai_search_query");

        function openPopover() {
            backdrop.classList.remove("hidden");
            if (queryInput) {
                queryInput.focus();
            }
        }

        function closePopover(restoreFocus) {
            backdrop.classList.add("hidden");
            if (restoreFocus) {
                openButton.focus();
            }
        }

        openButton.addEventListener("click", openPopover);
        cancelButton.addEventListener("click", function () {
            closePopover(true);
        });
        backdrop.addEventListener("click", function (event) {
            if (event.target === backdrop) {
                closePopover(false);
            }
        });
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !backdrop.classList.contains("hidden")) {
                closePopover(true);
            }
        });

        form.addEventListener("submit", function () {
            closePopover(false);
            if (window.WorkbenchLoading && typeof window.WorkbenchLoading.show === "function") {
                window.WorkbenchLoading.show("Screening reports...");
            }
        });
    }

    function setupDatasetPagination() {
        if (document.body.dataset.paginationBound === "1") {
            return;
        }
        document.body.dataset.paginationBound = "1";

        function buildUrlWithSharedQuery(baseUrl, sharedQuery) {
            if (!sharedQuery) {
                return baseUrl;
            }
            return `${baseUrl}${baseUrl.includes("?") ? "&" : "?"}${sharedQuery}`;
        }

        function replaceDatasetFromHtml(html, targetUrl) {
            const wrapper = document.createElement("div");
            wrapper.innerHTML = html.trim();
            const incomingDataset = wrapper.querySelector(".explore-surface--dataset.data-card");
            const currentDataset = document.querySelector("#page-content .explore-surface--dataset.data-card");

            if (!incomingDataset || !currentDataset) {
                window.location.href = targetUrl;
                return;
            }

            incomingDataset.classList.remove("reveal-up");
            incomingDataset.classList.add("is-visible");
            currentDataset.replaceWith(incomingDataset);
            window.history.replaceState(window.history.state, "", targetUrl);
            setupTableScrollbars();
            setupDatasetCellPreview();
            setupDatasetCollapse();

            if (typeof incomingDataset.animate === "function") {
                incomingDataset.animate(
                    [{ opacity: 0.9 }, { opacity: 1 }],
                    { duration: 90, easing: "linear" }
                );
            }
        }

        function fetchAndSwapDataset(panelUrl, targetUrl) {
            fetch(panelUrl, {
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                },
                credentials: "same-origin",
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Pagination request failed");
                    }
                    return response.text();
                })
                .then((html) => replaceDatasetFromHtml(html, targetUrl))
                .catch(() => {
                    window.location.href = targetUrl;
                });
        }

        window.WorkbenchDatasetPanel = {
            fetchAndSwapDataset: fetchAndSwapDataset,
        };

        document.addEventListener("click", function (event) {
            const link = event.target.closest(".dataset-pagination__actions a[data-dataset-panel-url]");
            if (!link) {
                return;
            }
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
                return;
            }

            const panelUrlRaw = link.getAttribute("data-dataset-panel-url");
            if (!panelUrlRaw) {
                return;
            }
            const panelUrl = new URL(panelUrlRaw, window.location.origin);
            const targetUrl = new URL(link.href, window.location.origin);
            if (targetUrl.origin !== window.location.origin) {
                return;
            }

            event.preventDefault();
            fetchAndSwapDataset(panelUrl.pathname + panelUrl.search, targetUrl.pathname + targetUrl.search);
        });

        document.addEventListener("submit", function (event) {
            const form = event.target.closest("form[data-dataset-goto]");
            if (!form) {
                return;
            }
            event.preventDefault();

            const input = form.querySelector("input[name='page']");
            if (!input) {
                return;
            }
            const max = Number(input.max || "1");
            const min = Number(input.min || "1");
            let page = Number(input.value || "1");
            if (!Number.isFinite(page)) {
                page = min;
            }
            page = Math.max(min, Math.min(max, Math.floor(page)));
            input.value = String(page);

            const panelBase = form.getAttribute("data-dataset-panel-base") || "/dataset-panel/";
            const browserBase = form.getAttribute("data-dataset-browser-base") || "?page=";
            const sharedQuery = (form.getAttribute("data-dataset-shared-query") || "").trim();
            const panelUrl = buildUrlWithSharedQuery(`${panelBase}?page=${page}`, sharedQuery);
            const targetUrl = buildUrlWithSharedQuery(`${browserBase}${page}`, sharedQuery);
            fetchAndSwapDataset(panelUrl, targetUrl);
        });
    }

    function setupDatasetCollapse() {
        const root = document.querySelector(".explore-surface--dataset[data-dataset-collapse-root]");
        if (!root) {
            return;
        }

        const toggle = root.querySelector("[data-dataset-collapse-toggle]");
        const label = root.querySelector("[data-dataset-collapse-label]");
        if (!toggle || !label) {
            return;
        }
        if (toggle.dataset.bound === "1") {
            return;
        }
        toggle.dataset.bound = "1";

        const storageKey = "workbench.datasetExpanded";
        let expanded = false;
        try {
            expanded = window.localStorage.getItem(storageKey) === "1";
        } catch (error) {
            expanded = false;
        }

        function paint(isExpanded) {
            root.classList.toggle("is-collapsed", !isExpanded);
            toggle.setAttribute("aria-expanded", isExpanded ? "true" : "false");
            label.textContent = isExpanded ? "Hide individual reports" : "Show individual reports";
        }

        paint(expanded);

        toggle.addEventListener("click", function () {
            expanded = !expanded;
            paint(expanded);
            try {
                window.localStorage.setItem(storageKey, expanded ? "1" : "0");
            } catch (error) {
                // Ignore storage access issues.
            }
        });
    }

    function setupExploreDashboard() {
        const root = byId("explore-dashboard");
        if (!root) {
            return;
        }

        const dataNode = byId("explore-dashboard-data");
        if (!dataNode) {
            return;
        }

        const resetButton = byId("dashboard-filter-reset");
        const statusNode = byId("dashboard-status");
        const statReports = byId("dashboard-stat-reports");
        const statCoroners = byId("dashboard-stat-coroners");
        const statReceiverLinks = byId("dashboard-stat-receiver-links");
        const searchCoroner = byId("dashboard-filter-coroner-search");
        const searchArea = byId("dashboard-filter-area-search");
        const searchReceiver = byId("dashboard-filter-receiver-search");
        const optionsCoroner = byId("dashboard-options-coroner");
        const optionsArea = byId("dashboard-options-area");
        const optionsReceiver = byId("dashboard-options-receiver");
        const badgesCoroner = byId("dashboard-badges-coroner");
        const badgesArea = byId("dashboard-badges-area");
        const badgesReceiver = byId("dashboard-badges-receiver");
        const filterCoronerField = root.querySelector("[data-dashboard-filter='coroner']");
        const filterAreaField = root.querySelector("[data-dashboard-filter='area']");
        const filterReceiverField = root.querySelector("[data-dashboard-filter='receiver']");

        const chartRoots = {
            monthly: byId("dashboard-chart-monthly"),
            coroner: byId("dashboard-chart-coroner"),
            area: byId("dashboard-chart-area"),
            receiver: byId("dashboard-chart-receiver"),
        };

        if (
            !resetButton ||
            !statReports ||
            !statCoroners ||
            !statReceiverLinks ||
            !searchCoroner ||
            !searchArea ||
            !searchReceiver ||
            !optionsCoroner ||
            !optionsArea ||
            !optionsReceiver ||
            !badgesCoroner ||
            !badgesArea ||
            !badgesReceiver ||
            !filterCoronerField ||
            !filterAreaField ||
            !filterReceiverField ||
            !chartRoots.monthly ||
            !chartRoots.coroner ||
            !chartRoots.area ||
            !chartRoots.receiver
        ) {
            return;
        }

        if (root.dataset.bound === "1") {
            if (window.echarts && typeof window.echarts.getInstanceByDom === "function") {
                Object.values(chartRoots).forEach((chartRoot) => {
                    const instance = window.echarts.getInstanceByDom(chartRoot);
                    if (instance) {
                        instance.resize();
                    }
                });
            }
            return;
        }
        root.dataset.bound = "1";

        function setStatus(text) {
            if (statusNode) {
                statusNode.textContent = text;
            }
        }

        let payload = {};
        try {
            payload = JSON.parse(dataNode.textContent || "{}");
        } catch (error) {
            setStatus("Dashboard data could not be parsed.");
            return;
        }

        if (!window.echarts || typeof window.echarts.init !== "function") {
            setStatus("ECharts is not available. Reload the page and try again.");
            return;
        }

        const rows = Array.isArray(payload.rows) ? payload.rows : [];
        const options = payload && typeof payload.options === "object" ? payload.options : {};
        const selected = payload && typeof payload.selected === "object" ? payload.selected : {};
        const coronerOptions = Array.isArray(options.coroners) ? options.coroners : [];
        const areaOptions = Array.isArray(options.areas) ? options.areas : [];
        const receiverOptions = Array.isArray(options.receivers) ? options.receivers : [];

        const TOP_N = 12;
        const UNKNOWN_LABEL = "Not specified";
        const gridTextColor = "rgba(209, 220, 255, 0.7)";
        const splitLineColor = "rgba(156, 173, 232, 0.18)";

        function normaliseValue(value) {
            if (typeof value !== "string") {
                if (value === null || value === undefined) {
                    return "";
                }
                return String(value).trim();
            }
            return value.trim();
        }

        function normaliseReceivers(rawReceivers) {
            if (!Array.isArray(rawReceivers) || !rawReceivers.length) {
                return [];
            }
            return rawReceivers
                .map((value) => normaliseValue(value))
                .filter((value) => Boolean(value));
        }

        const charts = {};
        Object.entries(chartRoots).forEach(([chartName, chartRoot]) => {
            const existingInstance = window.echarts.getInstanceByDom(chartRoot);
            if (existingInstance) {
                existingInstance.dispose();
            }
            charts[chartName] = window.echarts.init(chartRoot, null, { renderer: "canvas" });
        });

        const selectedFilters = {
            coroner: new Set(Array.isArray(selected.coroner) ? selected.coroner.map(normaliseValue).filter(Boolean) : []),
            area: new Set(Array.isArray(selected.area) ? selected.area.map(normaliseValue).filter(Boolean) : []),
            receiver: new Set(Array.isArray(selected.receiver) ? selected.receiver.map(normaliseValue).filter(Boolean) : []),
        };
        const filterFieldNodes = {
            coroner: filterCoronerField,
            area: filterAreaField,
            receiver: filterReceiverField,
        };
        let openFilterName = "";

        function setOpenFilter(fieldName) {
            openFilterName = fieldName || "";
            Object.entries(filterFieldNodes).forEach(([name, node]) => {
                node.classList.toggle("is-open", name === openFilterName);
            });
        }

        function syncBrowserUrl() {
            const params = new URLSearchParams(window.location.search);
            params.delete("page");
            ["coroner", "area", "receiver"].forEach((field) => {
                params.delete(field);
                Array.from(selectedFilters[field]).forEach((value) => params.append(field, value));
            });
            const query = params.toString();
            const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
            window.history.replaceState(window.history.state, "", nextUrl);
        }

        function buildFilterQueryString() {
            const params = new URLSearchParams();
            ["coroner", "area", "receiver"].forEach((field) => {
                Array.from(selectedFilters[field]).forEach((value) => params.append(field, value));
            });
            return params.toString();
        }

        function refreshDatasetTable() {
            if (!window.WorkbenchDatasetPanel || typeof window.WorkbenchDatasetPanel.fetchAndSwapDataset !== "function") {
                return;
            }
            const query = buildFilterQueryString();
            const panelUrl = `/dataset-panel/?page=1${query ? `&${query}` : ""}`;
            const targetUrl = `?page=1${query ? `&${query}` : ""}`;
            window.WorkbenchDatasetPanel.fetchAndSwapDataset(panelUrl, targetUrl);
        }

        function toSortedCountRows(counter) {
            return Array.from(counter.entries())
                .sort((left, right) => {
                    if (right[1] !== left[1]) {
                        return right[1] - left[1];
                    }
                    return left[0].localeCompare(right[0], undefined, { sensitivity: "base" });
                })
                .map(([name, value]) => ({ name, value }));
        }

        function filterRows(sourceRows) {
            return sourceRows.filter((row) => {
                const rowCoroner = normaliseValue(row.coroner);
                const rowArea = normaliseValue(row.area);
                const rowReceivers = normaliseReceivers(row.receivers);

                if (selectedFilters.coroner.size && !selectedFilters.coroner.has(rowCoroner)) {
                    return false;
                }
                if (selectedFilters.area.size && !selectedFilters.area.has(rowArea)) {
                    return false;
                }
                if (selectedFilters.receiver.size) {
                    const hasMatch = rowReceivers.some((value) => selectedFilters.receiver.has(value));
                    if (!hasMatch) {
                        return false;
                    }
                }
                if (selectedFilters.receiver.size && !rowReceivers.length) {
                    return false;
                }
                return true;
            });
        }

        function computeMonthlySeries(sourceRows) {
            const monthlyCounts = new Map();
            sourceRows.forEach((row) => {
                const month = normaliseValue(row.year_month);
                if (!month) {
                    return;
                }
                monthlyCounts.set(month, (monthlyCounts.get(month) || 0) + 1);
            });

            return Array.from(monthlyCounts.entries())
                .sort(([left], [right]) => left.localeCompare(right))
                .map(([name, value]) => ({ name, value }));
        }

        function computeTopValues(sourceRows, fieldName) {
            const counts = new Map();
            sourceRows.forEach((row) => {
                const value = normaliseValue(row[fieldName]) || UNKNOWN_LABEL;
                counts.set(value, (counts.get(value) || 0) + 1);
            });
            return toSortedCountRows(counts).slice(0, TOP_N);
        }

        function computeTopReceivers(sourceRows) {
            const counts = new Map();
            sourceRows.forEach((row) => {
                const receiverList = normaliseReceivers(row.receivers);
                if (!receiverList.length) {
                    counts.set(UNKNOWN_LABEL, (counts.get(UNKNOWN_LABEL) || 0) + 1);
                    return;
                }
                receiverList.forEach((receiverName) => {
                    counts.set(receiverName, (counts.get(receiverName) || 0) + 1);
                });
            });
            return toSortedCountRows(counts).slice(0, TOP_N);
        }

        function renderMonthlyChart(seriesRows) {
            const months = seriesRows.map((row) => row.name);
            const counts = seriesRows.map((row) => row.value);
            charts.monthly.setOption(
                {
                    animationDuration: 240,
                    tooltip: { trigger: "axis" },
                    grid: { left: 44, right: 12, top: 24, bottom: 28 },
                    xAxis: {
                        type: "category",
                        data: months,
                        axisLabel: { color: gridTextColor, rotate: 30 },
                        axisLine: { lineStyle: { color: splitLineColor } },
                    },
                    yAxis: {
                        type: "value",
                        axisLabel: { color: gridTextColor },
                        splitLine: { lineStyle: { color: splitLineColor } },
                    },
                    series: [
                        {
                            type: "line",
                            smooth: true,
                            showSymbol: false,
                            lineStyle: { width: 2.2, color: "#66d7ff" },
                            itemStyle: { color: "#66d7ff" },
                            areaStyle: { color: "rgba(102, 215, 255, 0.18)" },
                            data: counts,
                        },
                    ],
                    graphic: !seriesRows.length
                        ? [
                            {
                                type: "text",
                                left: "center",
                                top: "middle",
                                style: {
                                    text: "No dated reports for this filter.",
                                    fill: "rgba(210, 220, 255, 0.7)",
                                    fontSize: 13,
                                },
                            },
                        ]
                        : [],
                },
                true
            );
        }

        function renderBarChart(chart, seriesRows, colorHex, emptyText) {
            const labels = seriesRows.map((row) => row.name);
            const values = seriesRows.map((row) => row.value);
            chart.setOption(
                {
                    animationDuration: 240,
                    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
                    grid: { left: 140, right: 16, top: 24, bottom: 24 },
                    xAxis: {
                        type: "value",
                        axisLabel: { color: gridTextColor },
                        splitLine: { lineStyle: { color: splitLineColor } },
                    },
                    yAxis: {
                        type: "category",
                        inverse: true,
                        data: labels,
                        axisLabel: {
                            color: gridTextColor,
                            overflow: "truncate",
                            width: 130,
                        },
                        axisLine: { lineStyle: { color: splitLineColor } },
                    },
                    series: [
                        {
                            type: "bar",
                            barMaxWidth: 22,
                            data: values,
                            itemStyle: {
                                color: colorHex,
                                borderRadius: [0, 6, 6, 0],
                            },
                        },
                    ],
                    graphic: !seriesRows.length
                        ? [
                            {
                                type: "text",
                                left: "center",
                                top: "middle",
                                style: {
                                    text: emptyText,
                                    fill: "rgba(210, 220, 255, 0.7)",
                                    fontSize: 13,
                                },
                            },
                        ]
                        : [],
                },
                true
            );
        }

        function updateStatCards(sourceRows) {
            const uniqueCoroners = new Set();
            let receiverLinks = 0;

            sourceRows.forEach((row) => {
                const coroner = normaliseValue(row.coroner);
                if (coroner) {
                    uniqueCoroners.add(coroner);
                }
                const receiverList = normaliseReceivers(row.receivers);
                receiverLinks += receiverList.length;
            });

            statReports.textContent = sourceRows.length.toLocaleString();
            statCoroners.textContent = uniqueCoroners.size.toLocaleString();
            statReceiverLinks.textContent = receiverLinks.toLocaleString();
        }

        function renderDashboard() {
            const visibleRows = filterRows(rows);
            updateStatCards(visibleRows);

            const monthlyRows = computeMonthlySeries(visibleRows);
            const topCoroners = computeTopValues(visibleRows, "coroner");
            const topAreas = computeTopValues(visibleRows, "area");
            const topReceivers = computeTopReceivers(visibleRows);

            renderMonthlyChart(monthlyRows);
            renderBarChart(
                charts.coroner,
                topCoroners,
                "#5b9dff",
                "No coroner values for this filter."
            );
            renderBarChart(
                charts.area,
                topAreas,
                "#7ac26b",
                "No area values for this filter."
            );
            renderBarChart(
                charts.receiver,
                topReceivers,
                "#f2a85a",
                "No receiver values for this filter."
            );

            setStatus(`Showing ${visibleRows.length.toLocaleString()} of ${rows.length.toLocaleString()} reports.`);
        }

        function renderFilterControl(fieldName, allOptions, searchInput, optionsNode, badgesNode) {
            function createBadge(value) {
                const badge = document.createElement("button");
                badge.type = "button";
                badge.className = "dashboard-selection-badge";
                badge.title = `Remove ${value}`;
                badge.innerHTML = `<span>${value}</span><i aria-hidden="true">x</i>`;
                badge.addEventListener("click", function () {
                    selectedFilters[fieldName].delete(value);
                    renderDashboard();
                    renderFilterControl(fieldName, allOptions, searchInput, optionsNode, badgesNode);
                    syncBrowserUrl();
                    refreshDatasetTable();
                });
                return badge;
            }

            function createOption(value) {
                const selectedNow = selectedFilters[fieldName].has(value);
                const optionButton = document.createElement("button");
                optionButton.type = "button";
                optionButton.className = `dashboard-option-item${selectedNow ? " is-selected" : ""}`;
                optionButton.innerHTML = `<span>${value}</span><strong>${selectedNow ? "Selected" : "Add"}</strong>`;
                optionButton.addEventListener("click", function () {
                    if (selectedNow) {
                        selectedFilters[fieldName].delete(value);
                    } else {
                        selectedFilters[fieldName].add(value);
                    }
                    renderDashboard();
                    renderFilterControl(fieldName, allOptions, searchInput, optionsNode, badgesNode);
                    syncBrowserUrl();
                    refreshDatasetTable();
                });
                return optionButton;
            }

            function paintBadges() {
                badgesNode.innerHTML = "";
                const selectedValues = Array.from(selectedFilters[fieldName]).sort((a, b) =>
                    a.localeCompare(b, undefined, { sensitivity: "base" })
                );
                if (!selectedValues.length) {
                    const empty = document.createElement("span");
                    empty.className = "dashboard-filter-empty";
                    empty.textContent = "All selected";
                    badgesNode.appendChild(empty);
                    return;
                }
                selectedValues.forEach((value) => badgesNode.appendChild(createBadge(value)));
            }

            function paintOptions() {
                optionsNode.innerHTML = "";
                const searchTerm = normaliseValue(searchInput.value).toLowerCase();
                const filteredOptions = allOptions.filter((value) => {
                    if (!searchTerm) {
                        return true;
                    }
                    return value.toLowerCase().includes(searchTerm);
                });
                if (!filteredOptions.length) {
                    const empty = document.createElement("div");
                    empty.className = "dashboard-filter-empty";
                    empty.textContent = "No matches";
                    optionsNode.appendChild(empty);
                    return;
                }
                filteredOptions.slice(0, 120).forEach((value) => {
                    optionsNode.appendChild(createOption(value));
                });
            }

            paintBadges();
            paintOptions();
        }

        function resetFilters() {
            selectedFilters.coroner.clear();
            selectedFilters.area.clear();
            selectedFilters.receiver.clear();
            searchCoroner.value = "";
            searchArea.value = "";
            searchReceiver.value = "";
            renderDashboard();
            renderFilterControl("coroner", coronerOptions, searchCoroner, optionsCoroner, badgesCoroner);
            renderFilterControl("area", areaOptions, searchArea, optionsArea, badgesArea);
            renderFilterControl("receiver", receiverOptions, searchReceiver, optionsReceiver, badgesReceiver);
            syncBrowserUrl();
            refreshDatasetTable();
        }

        [
            { field: "coroner", options: coronerOptions, search: searchCoroner, list: optionsCoroner, badges: badgesCoroner },
            { field: "area", options: areaOptions, search: searchArea, list: optionsArea, badges: badgesArea },
            { field: "receiver", options: receiverOptions, search: searchReceiver, list: optionsReceiver, badges: badgesReceiver },
        ].forEach((config) => {
            const containerNode = filterFieldNodes[config.field];
            containerNode.addEventListener("click", function () {
                setOpenFilter(config.field);
            });
            config.search.addEventListener("input", function () {
                renderFilterControl(config.field, config.options, config.search, config.list, config.badges);
            });
            config.search.addEventListener("focus", function () {
                setOpenFilter(config.field);
            });
        });

        document.addEventListener("click", function (event) {
            if (!openFilterName) {
                return;
            }
            const openFieldNode = filterFieldNodes[openFilterName];
            if (!openFieldNode) {
                setOpenFilter("");
                return;
            }
            if (!openFieldNode.contains(event.target)) {
                setOpenFilter("");
            }
        });
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                setOpenFilter("");
            }
        });

        resetButton.addEventListener("click", resetFilters);
        window.addEventListener("resize", function () {
            Object.values(charts).forEach((chart) => chart.resize());
        });

        renderFilterControl("coroner", coronerOptions, searchCoroner, optionsCoroner, badgesCoroner);
        renderFilterControl("area", areaOptions, searchArea, optionsArea, badgesArea);
        renderFilterControl("receiver", receiverOptions, searchReceiver, optionsReceiver, badgesReceiver);
        renderDashboard();
    }

    function setupConfigModalDismiss() {
        const modal = byId("config-modal");
        if (!modal) {
            return;
        }
        if (modal.dataset.bound === "1") {
            return;
        }
        modal.dataset.bound = "1";

        const cancelButton = modal.querySelector("[data-config-cancel]");
        document.body.classList.add("modal-open");

        function closeModal() {
            modal.classList.add("hidden");
            document.body.classList.remove("modal-open");
        }

        if (cancelButton) {
            cancelButton.addEventListener("click", closeModal);
        }

        modal.addEventListener("click", function (event) {
            if (event.target === modal) {
                closeModal();
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !modal.classList.contains("hidden")) {
                closeModal();
            }
        });
    }

    function setupGlobalLoadingOverlay() {
        if (document.body.dataset.globalLoadingBound === "1") {
            return;
        }
        document.body.dataset.globalLoadingBound = "1";

        const overlay = byId("global-loading-overlay");
        const messageNode = byId("global-loading-message");
        if (!overlay || !messageNode) {
            return;
        }

        const LONG_RUNNING_ACTIONS = new Set([
            "load_reports",
            "filter_reports",
            "discover_themes",
            "extract_features",
        ]);

        const ACTION_MESSAGES = {
            load_reports: "Loading report data and preparing your workspace...",
            filter_reports: "Screening reports...",
            discover_themes: "Discovering themes and building a preview...",
            extract_features: "Extracting structured fields...",
        };

        function showLoader(message) {
            messageNode.textContent = message || "Running your request...";
            overlay.classList.add("is-active");
            overlay.setAttribute("aria-hidden", "false");
            document.body.classList.add("loading-active");
        }

        function hideLoader() {
            overlay.classList.remove("is-active");
            overlay.setAttribute("aria-hidden", "true");
            document.body.classList.remove("loading-active");
        }

        function getSubmittedAction(form, submitter) {
            if (submitter && submitter.name === "action") {
                return (submitter.value || "").trim();
            }
            const directAction = form.querySelector("input[name='action']");
            if (directAction) {
                return (directAction.value || "").trim();
            }
            return "";
        }

        document.addEventListener("submit", function (event) {
            const form = event.target.closest("form");
            if (!form || event.defaultPrevented) {
                return;
            }
            if (form.hasAttribute("data-dataset-goto")) {
                return;
            }

            const action = getSubmittedAction(form, event.submitter);
            if (!LONG_RUNNING_ACTIONS.has(action)) {
                return;
            }

            showLoader(ACTION_MESSAGES[action] || "Running your request...");
        }, true);

        window.addEventListener("pageshow", hideLoader);
        document.addEventListener("visibilitychange", function () {
            if (document.visibilityState === "visible") {
                hideLoader();
            }
        });

        window.WorkbenchLoading = {
            show: showLoader,
            hide: hideLoader,
        };
    }

    function initializePageFeatures() {
        setupGlobalLoadingOverlay();
        setupTableScrollbars();
        setupPageScrollbar();
        toggleProviderFields();
        toggleDiscoverTrimFields();
        toggleTruncationTypeFields();
        setupFeatureGrid();
        setupDatasetCollapse();
        setupDatasetCellPreview();
        setupRevealAnimations();
        setupStartOverConfirm();
        setupDownloadBundleModal();
        setupTopbarQuickSettings();
        setupTopbarLimitControls();
        setupTopbarLLMProviderToggle();
        setupTopbarAIResetConfirm();
        setupTopbarDateValidation();
        setupAdvancedAIFilterPopover();
        setupDatasetPagination();
        setupExploreDashboard();
        setupConfigModalDismiss();

        const provider = byId("provider_override");
        if (provider && provider.dataset.bound !== "1") {
            provider.dataset.bound = "1";
            provider.addEventListener("change", toggleProviderFields);
        }

        const trimApproach = byId("trim_approach");
        if (trimApproach && trimApproach.dataset.bound !== "1") {
            trimApproach.dataset.bound = "1";
            trimApproach.addEventListener("change", toggleDiscoverTrimFields);
        }

        const truncType = byId("truncation_limit_type");
        if (truncType && truncType.dataset.bound !== "1") {
            truncType.dataset.bound = "1";
            truncType.addEventListener("change", toggleTruncationTypeFields);
        }
    }

    function setupClientNavigation() {
        const contentRoot = byId("page-content");
        if (!contentRoot) {
            return;
        }

        function getWorkspaceToken(rootNode) {
            if (!rootNode || !rootNode.dataset) {
                return "";
            }
            return rootNode.dataset.workspaceToken || "";
        }

        const pageCache = new Map();
        let activePath = window.location.pathname;
        let activeWorkspaceToken = getWorkspaceToken(contentRoot);
        if (contentRoot.firstElementChild) {
            pageCache.set(activePath, {
                node: contentRoot.firstElementChild,
                workspaceToken: activeWorkspaceToken,
            });
        }

        function swapTo(pathname, page, node, pushHistory, workspaceToken) {
            if (!node) {
                return;
            }
            contentRoot.replaceChildren(node);
            contentRoot.dataset.page = page;
            contentRoot.dataset.workspaceToken = workspaceToken || "";
            applyBodyPageClass(page);
            syncSidebarActive(page);
            if (pushHistory) {
                window.history.pushState(
                    { path: pathname, page: page, workspaceToken: workspaceToken || "" },
                    "",
                    pathname
                );
            }
            activePath = pathname;
            activeWorkspaceToken = workspaceToken || "";
            initializePageFeatures();
            if (typeof contentRoot.animate === "function") {
                contentRoot.animate(
                    [{ opacity: 0.82 }, { opacity: 1 }],
                    { duration: 140, easing: "ease-out" }
                );
            }
            window.scrollTo({ top: 0, left: 0, behavior: "auto" });
        }

        document.addEventListener("click", function (event) {
            const link = event.target.closest(".sidebar-nav a[data-page-link]");
            if (!link) {
                return;
            }
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
                return;
            }

            const targetUrl = new URL(link.href, window.location.origin);
            if (targetUrl.origin !== window.location.origin) {
                return;
            }

            const targetPath = targetUrl.pathname;
            const currentPath = window.location.pathname;
            if (targetPath === currentPath) {
                event.preventDefault();
                return;
            }

            event.preventDefault();
            const currentNode = contentRoot.firstElementChild;
            if (currentNode) {
                pageCache.set(currentPath, {
                    node: currentNode,
                    workspaceToken: activeWorkspaceToken,
                });
            }

            const targetPage = link.getAttribute("data-page-link") || getPageFromPath(targetPath);
            const cachedEntry = pageCache.get(targetPath);
            if (cachedEntry && cachedEntry.workspaceToken === activeWorkspaceToken) {
                swapTo(targetPath, targetPage, cachedEntry.node, true, cachedEntry.workspaceToken);
                return;
            }
            if (cachedEntry) {
                pageCache.delete(targetPath);
            }

            fetch(targetPath, {
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                },
                credentials: "same-origin",
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Navigation request failed");
                    }
                    return response.text();
                })
                .then((html) => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, "text/html");
                    const incomingRoot = doc.querySelector("#page-content");
                    const incomingNode = incomingRoot ? incomingRoot.firstElementChild : null;
                    const page = incomingRoot && incomingRoot.dataset.page
                        ? incomingRoot.dataset.page
                        : targetPage;
                    const workspaceToken = incomingRoot ? getWorkspaceToken(incomingRoot) : "";
                    if (!incomingNode) {
                        window.location.href = targetPath;
                        return;
                    }
                    pageCache.set(targetPath, {
                        node: incomingNode,
                        workspaceToken: workspaceToken,
                    });
                    swapTo(targetPath, page, incomingNode, true, workspaceToken);
                })
                .catch(() => {
                    window.location.href = targetPath;
                });
        });

        window.addEventListener("popstate", function () {
            const path = window.location.pathname;
            const page = getPageFromPath(path);
            const currentNode = contentRoot.firstElementChild;
            if (currentNode) {
                pageCache.set(activePath, {
                    node: currentNode,
                    workspaceToken: activeWorkspaceToken,
                });
            }
            const cachedEntry = pageCache.get(path);
            if (cachedEntry && cachedEntry.workspaceToken === activeWorkspaceToken) {
                swapTo(path, page, cachedEntry.node, false, cachedEntry.workspaceToken);
                return;
            }
            window.location.reload();
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        setupSidebarCollapse();
        setupClientNavigation();
        initializePageFeatures();
    });
})();
