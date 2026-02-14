(function () {
    document.documentElement.classList.add("js-enabled");
    const PAGE_CLASS_PREFIX = "page-";
    const KNOWN_PAGES = ["home", "explore", "themes", "extract", "for_coders", "settings"];

    function byId(id) {
        return document.getElementById(id);
    }

    function getCssVar(name, fallback) {
        const value = window.getComputedStyle(document.body).getPropertyValue(name);
        const trimmed = value ? value.trim() : "";
        return trimmed || fallback;
    }

    function getCsrfTokenFromPage() {
        const cookie = document.cookie || "";
        const match = cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
        if (match && match[1]) {
            return decodeURIComponent(match[1]);
        }
        const csrfInput = document.querySelector("input[name='csrfmiddlewaretoken']");
        return csrfInput ? csrfInput.value : "";
    }

    function getPageFromPath(pathname) {
        if (pathname.startsWith("/explore-pfds")) {
            return "explore";
        }
        if (pathname.startsWith("/workbooks/")) {
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
        if (pathname.startsWith("/for-coders")) {
            return "for_coders";
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
        const datasetTables = Array.from(document.querySelectorAll(".data-card .data-table"));
        if (!datasetTables.length) {
            return;
        }

        function getColumnClass(headerText) {
            const label = (headerText || "").trim().toLowerCase();
            if (label === "actions" || label === "restore") {
                return "dataset-col-action";
            }
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

        const previewCells = [];
        datasetTables.forEach((datasetTable) => {
            if (datasetTable.dataset.previewBound === "1") {
                return;
            }
            datasetTable.dataset.previewBound = "1";

            const headerCells = Array.from(datasetTable.querySelectorAll("thead th"));
            const bodyRows = Array.from(datasetTable.querySelectorAll("tbody tr"));
            if (bodyRows.length > 120 || !headerCells.length) {
                return;
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

            const cells = datasetTable.querySelectorAll(
                "tbody td:not(.dataset-row-actions-cell):not(.dataset-row-restore-cell)"
            );
            cells.forEach((cell) => previewCells.push(cell));
        });

        if (!previewCells.length) {
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

        previewCells.forEach((cell) => {
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
            const clickedCell = event.target.closest(
                ".data-card .data-table tbody td:not(.dataset-row-actions-cell):not(.dataset-row-restore-cell)"
            );
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

        const datasetWraps = document.querySelectorAll(".data-card .table-wrap");
        datasetWraps.forEach((datasetWrap) => {
            const viewport = datasetWrap.querySelector("[data-overlayscrollbars-viewport]");
            if (viewport) {
                viewport.addEventListener("scroll", hidePopup, { passive: true });
            }
            datasetWrap.addEventListener("scroll", hidePopup, { passive: true });
        });

        window.addEventListener("resize", hidePopup);
    }

    function setupDatasetRowActions() {
        function closeAllRowPopovers(exceptPopover) {
            document.querySelectorAll("[data-row-delete-popover]").forEach((popover) => {
                if (exceptPopover && popover === exceptPopover) {
                    return;
                }
                const root = popover.closest("[data-row-actions-root]");
                popover.classList.add("hidden");
                if (root) {
                    root.classList.remove("is-open");
                }
            });
        }

        if (document.body.dataset.datasetRowActionsBound === "1") {
            return;
        }
        document.body.dataset.datasetRowActionsBound = "1";

        function openPopoverForRoot(root) {
            if (!root) {
                return;
            }
            const popover = root.querySelector("[data-row-delete-popover]");
            if (!popover) {
                return;
            }
            closeAllRowPopovers(popover);
            popover.classList.remove("hidden");
            root.classList.add("is-open");
            const reasonInput = root.querySelector("textarea[name='exclusion_reason']");
            if (reasonInput) {
                reasonInput.focus();
            }
        }

        function closePopoverForRoot(root, restoreFocus) {
            if (!root) {
                return;
            }
            const popover = root.querySelector("[data-row-delete-popover]");
            const toggle = root.querySelector("[data-row-delete-toggle]");
            if (popover) {
                popover.classList.add("hidden");
            }
            root.classList.remove("is-open");
            if (restoreFocus && toggle) {
                toggle.focus();
            }
        }

        document.addEventListener("click", function (event) {
            const toggle = event.target.closest("[data-row-delete-toggle]");
            if (toggle) {
                const root = toggle.closest("[data-row-actions-root]");
                if (!root) {
                    return;
                }
                event.preventDefault();
                event.stopPropagation();
                const popover = root.querySelector("[data-row-delete-popover]");
                if (!popover) {
                    return;
                }
                if (popover.classList.contains("hidden")) {
                    openPopoverForRoot(root);
                } else {
                    closePopoverForRoot(root, true);
                }
                return;
            }

            const cancelButton = event.target.closest("[data-row-delete-cancel]");
            if (cancelButton) {
                const root = cancelButton.closest("[data-row-actions-root]");
                event.preventDefault();
                event.stopPropagation();
                closePopoverForRoot(root, true);
                return;
            }

            if (event.target.closest("[data-row-delete-popover]")) {
                event.stopPropagation();
            }
        });

        document.addEventListener("pointerdown", function (event) {
            if (event.target.closest("[data-row-actions-root]")) {
                return;
            }
            closeAllRowPopovers();
        }, true);

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closeAllRowPopovers();
            }
        });
    }

    function setupDatasetMutations() {
        if (document.body.dataset.datasetMutationsBound === "1") {
            return;
        }
        document.body.dataset.datasetMutationsBound = "1";

        let inFlight = false;

        function applyPageContentFromHtml(html, responseUrl) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, "text/html");
            const incomingRoot = doc.querySelector("#page-content");
            const currentRoot = byId("page-content");
            if (!incomingRoot || !currentRoot) {
                return false;
            }

            currentRoot.innerHTML = incomingRoot.innerHTML;
            currentRoot.dataset.page = incomingRoot.dataset.page || "explore";
            currentRoot.dataset.workspaceToken = incomingRoot.dataset.workspaceToken || "";

            const page = currentRoot.dataset.page || getPageFromPath(window.location.pathname);
            applyBodyPageClass(page);
            syncSidebarActive(page);

            if (responseUrl) {
                try {
                    const target = new URL(responseUrl, window.location.origin);
                    window.history.replaceState(window.history.state, "", target.pathname + target.search);
                } catch (error) {
                    // Ignore URL parse issues.
                }
            }

            initializePageFeatures();
            return true;
        }

        document.addEventListener("submit", function (event) {
            const form = event.target.closest("form[data-dataset-mutate]");
            if (!form) {
                return;
            }
            event.preventDefault();

            if (inFlight) {
                return;
            }
            inFlight = true;

            const submitButton = event.submitter;
            if (submitButton) {
                submitButton.disabled = true;
            }

            const targetUrl = form.getAttribute("action") || window.location.href;
            const body = new FormData(form);

            fetch(targetUrl, {
                method: "POST",
                body: body,
                credentials: "same-origin",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                },
            })
                .then((response) => response.text().then((text) => ({ response, text })))
                .then(({ response, text }) => {
                    if (!response.ok) {
                        throw new Error("Dataset update failed");
                    }
                    const swapped = applyPageContentFromHtml(text, response.url);
                    if (!swapped) {
                        throw new Error("Dataset update response was invalid");
                    }
                })
                .catch(() => {
                    window.location.href = targetUrl;
                })
                .finally(() => {
                    inFlight = false;
                    if (submitButton) {
                        submitButton.disabled = false;
                    }
                });
        }, true);
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

    function setupForCodersLinkRouting() {
        if (document.body.dataset.forCodersRoutingBound === "1") {
            return;
        }
        document.body.dataset.forCodersRoutingBound = "1";

        document.addEventListener("click", function (event) {
            const link = event.target.closest(".coder-docs-article a, .coder-docs-nav-link, .coder-docs-toc-item a");
            if (!link) {
                return;
            }
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
                return;
            }

            const href = link.getAttribute("href");
            if (!href || href.startsWith("#")) {
                return;
            }
            if (link.hasAttribute("download")) {
                return;
            }

            const targetAttr = (link.getAttribute("target") || "").toLowerCase();
            if (targetAttr && targetAttr !== "_self") {
                return;
            }

            let targetUrl = null;
            try {
                targetUrl = new URL(href, window.location.origin);
            } catch (error) {
                return;
            }
            if (!targetUrl || targetUrl.origin !== window.location.origin) {
                return;
            }
            if (!targetUrl.pathname.startsWith("/for-coders")) {
                return;
            }

            event.preventDefault();
            window.location.assign(targetUrl.pathname + targetUrl.search + targetUrl.hash);
        }, true);
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
        if (!document.querySelector(".topbar-popover-backdrop")) {
            return;
        }
        if (document.body.dataset.topbarQuickSettingsBound === "1") {
            return;
        }
        document.body.dataset.topbarQuickSettingsBound = "1";

        let active = null;

        function getPopoverByKey(key) {
            if (key === "report-limit") {
                return byId("report-limit-popover");
            }
            if (key === "date-range") {
                return byId("date-range-popover");
            }
            if (key === "llm-settings") {
                return byId("llm-settings-popover");
            }
            return null;
        }

        function positionPopover(popover, button) {
            const panel = popover.querySelector(".topbar-popover");
            if (!panel) {
                return;
            }

            const margin = 10;
            const buttonRect = button.getBoundingClientRect();
            const panelRect = panel.getBoundingClientRect();

            let left = buttonRect.left;
            if (left + panelRect.width > window.innerWidth - margin) {
                left = window.innerWidth - panelRect.width - margin;
            }
            left = Math.max(margin, left);

            let top = buttonRect.bottom + 10;
            if (top + panelRect.height > window.innerHeight - margin) {
                top = buttonRect.top - panelRect.height - 10;
            }
            top = Math.max(margin, top);

            popover.style.left = `${left}px`;
            popover.style.top = `${top}px`;
        }

        function closeAllPopovers(restoreFocusTarget) {
            document.querySelectorAll(".topbar-popover-backdrop").forEach((popover) => {
                popover.classList.add("hidden");
                popover.style.left = "";
                popover.style.top = "";
            });
            active = null;
            if (restoreFocusTarget) {
                restoreFocusTarget.focus();
            }
        }

        function isActivePopoverVisible() {
            return Boolean(
                active &&
                active.popover &&
                active.button &&
                active.button.isConnected &&
                !active.popover.classList.contains("hidden")
            );
        }

        document.addEventListener("click", function (event) {
            const openButton = event.target.closest("[data-topbar-open]");
            if (openButton) {
                const key = openButton.getAttribute("data-topbar-open");
                const targetPopover = getPopoverByKey(key);
                if (!targetPopover) {
                    return;
                }
                if (
                    active &&
                    active.popover === targetPopover &&
                    !targetPopover.classList.contains("hidden")
                ) {
                    closeAllPopovers(openButton);
                    return;
                }
                closeAllPopovers();
                targetPopover.classList.remove("hidden");
                positionPopover(targetPopover, openButton);
                active = { popover: targetPopover, button: openButton };
                return;
            }

            const closeButton = event.target.closest("[data-topbar-close]");
            if (closeButton) {
                closeAllPopovers();
            }
        });

        document.addEventListener("pointerdown", function (event) {
            if (!active || !active.popover || active.popover.classList.contains("hidden")) {
                return;
            }
            if (active.popover.contains(event.target) || active.button.contains(event.target)) {
                return;
            }
            closeAllPopovers();
        }, true);

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                const focusTarget = active ? active.button : null;
                closeAllPopovers(focusTarget);
            }
        });

        window.addEventListener("resize", function () {
            if (isActivePopoverVisible()) {
                positionPopover(active.popover, active.button);
            }
        });

        window.addEventListener("scroll", function () {
            if (isActivePopoverVisible()) {
                positionPopover(active.popover, active.button);
            }
        }, { passive: true, capture: true });
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
            const max = Number(slider.max || 7000);
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

    function setupSettingsLoadReportsConfirm() {
        const form = document.querySelector(".settings-workbench .config-form");
        const openButton = document.querySelector("[data-settings-load-open]");
        const confirmModal = byId("settings-load-reports-confirm");
        if (!form || !openButton) {
            return;
        }
        if (form.dataset.loadConfirmBound === "1") {
            return;
        }
        form.dataset.loadConfirmBound = "1";

        if (!confirmModal) {
            openButton.addEventListener("click", function (event) {
                if (form.dataset.loadReportsConfirmed === "1") {
                    return;
                }
                const allowed = window.confirm(
                    "Loading reports will refresh your existing workspace. Any unsaved workflow changes and generated outputs will be lost. Continue?"
                );
                if (!allowed) {
                    event.preventDefault();
                    return;
                }
                form.dataset.loadReportsConfirmed = "1";
            });
            return;
        }

        const cancelButton = confirmModal.querySelector("[data-settings-load-cancel]");
        const confirmButton = confirmModal.querySelector("[data-settings-load-confirm]");
        let pendingSubmitter = null;

        function closeConfirm(restoreFocus) {
            confirmModal.classList.add("hidden");
            document.body.classList.remove("modal-open");
            if (restoreFocus && pendingSubmitter && typeof pendingSubmitter.focus === "function") {
                pendingSubmitter.focus();
            }
            pendingSubmitter = null;
        }

        function openConfirm(submitter) {
            pendingSubmitter = submitter;
            confirmModal.classList.remove("hidden");
            document.body.classList.add("modal-open");
            if (cancelButton) {
                cancelButton.focus();
            }
        }

        if (openButton) {
            openButton.addEventListener("click", function (event) {
                if (form.dataset.loadReportsConfirmed === "1") {
                    return;
                }
                event.preventDefault();
                openConfirm(openButton);
            });
        }

        form.addEventListener("submit", function (event) {
            const submitter = event.submitter;
            const actionValue = submitter ? submitter.value : "";
            if (actionValue !== "load_reports") {
                return;
            }
            if (form.dataset.loadReportsConfirmed === "1") {
                delete form.dataset.loadReportsConfirmed;
                return;
            }
            event.preventDefault();
            openConfirm(submitter);
        });

        if (cancelButton) {
            cancelButton.addEventListener("click", function () {
                closeConfirm(true);
            });
        }

        if (confirmButton) {
            confirmButton.addEventListener("click", function () {
                if (!pendingSubmitter) {
                    closeConfirm(false);
                    return;
                }
                form.dataset.loadReportsConfirmed = "1";
                const submitter = pendingSubmitter;
                closeConfirm(false);
                if (typeof form.requestSubmit === "function") {
                    form.requestSubmit(submitter);
                    return;
                }
                submitter.click();
            });
        }

        confirmModal.addEventListener("click", function (event) {
            if (event.target === confirmModal) {
                closeConfirm(true);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !confirmModal.classList.contains("hidden")) {
                closeConfirm(true);
            }
        });
    }

    function setupAdvancedAIPopovers() {
        const confirmBackdrop = byId("advanced-ai-manual-filter-confirm-popover");
        const confirmList = byId("advanced-ai-manual-filter-list");
        const confirmMessage = byId("advanced-ai-manual-filter-confirm-message");
        const confirmCancel = document.querySelector("[data-advanced-ai-manual-filter-cancel]");
        const confirmKeep = document.querySelector("[data-advanced-ai-manual-filter-keep]");
        const confirmDiscard = document.querySelector("[data-advanced-ai-manual-filter-discard]");
        const themeRerunBackdrop = byId("advanced-ai-theme-rerun-confirm-popover");
        const themeRerunCancel = document.querySelector("[data-advanced-ai-theme-rerun-cancel]");
        const themeRerunConfirm = document.querySelector("[data-advanced-ai-theme-rerun-confirm]");
        let pendingManualFilterConfirm = null;
        let pendingThemeRerunConfirm = null;

        function getDashboardManualFilters() {
            if (
                window.WorkbenchDashboardFilters &&
                typeof window.WorkbenchDashboardFilters.getSelection === "function"
            ) {
                return window.WorkbenchDashboardFilters.getSelection();
            }
            return { coroner: [], area: [], receiver: [] };
        }

        function hasDashboardManualFilters(filters) {
            return Boolean(
                (Array.isArray(filters.coroner) && filters.coroner.length) ||
                (Array.isArray(filters.area) && filters.area.length) ||
                (Array.isArray(filters.receiver) && filters.receiver.length)
            );
        }

        function clearManualFilterHiddenInputs(form) {
            form.querySelectorAll("input[data-manual-filter-hidden='1']").forEach((input) => input.remove());
        }

        function setManualFilterHiddenInputs(form, filters) {
            clearManualFilterHiddenInputs(form);
            const mapping = [
                ["dashboard_coroner", Array.isArray(filters.coroner) ? filters.coroner : []],
                ["dashboard_area", Array.isArray(filters.area) ? filters.area : []],
                ["dashboard_receiver", Array.isArray(filters.receiver) ? filters.receiver : []],
            ];
            mapping.forEach(([name, values]) => {
                values.forEach((value) => {
                    const input = document.createElement("input");
                    input.type = "hidden";
                    input.name = name;
                    input.value = value;
                    input.setAttribute("data-manual-filter-hidden", "1");
                    form.appendChild(input);
                });
            });
        }

        function hasExistingThemes(form) {
            return form && form.dataset && form.dataset.hasExistingThemes === "1";
        }

        function getThemeRerunInput(form) {
            return form ? form.querySelector("input[data-confirm-rerun-themes]") : null;
        }

        function openManualFilterConfirm(config) {
            if (!confirmBackdrop || !confirmList || !confirmMessage) {
                return false;
            }
            if (window.WorkbenchLoading && typeof window.WorkbenchLoading.hide === "function") {
                window.WorkbenchLoading.hide();
            }
            pendingManualFilterConfirm = config;
            confirmList.innerHTML = "";

            const filters = config.filters || {};
            const segments = [
                ["Coroner", Array.isArray(filters.coroner) ? filters.coroner : []],
                ["Area", Array.isArray(filters.area) ? filters.area : []],
                ["Receiver", Array.isArray(filters.receiver) ? filters.receiver : []],
            ];
            segments.forEach(([label, values]) => {
                if (!values.length) {
                    return;
                }
                const item = document.createElement("li");
                item.textContent = `${label}: ${values.join(", ")}`;
                confirmList.appendChild(item);
            });

            confirmMessage.textContent = `You've applied manual filters to your reports. Do you want to ${config.actionLabel} for your current selection, or discard these manual filters first?`;
            if (config.sourceBackdrop) {
                config.sourceBackdrop.classList.add("hidden");
            }
            confirmBackdrop.classList.remove("hidden");
            return true;
        }

        function openThemeRerunConfirm(config) {
            if (!themeRerunBackdrop) {
                return false;
            }
            if (window.WorkbenchLoading && typeof window.WorkbenchLoading.hide === "function") {
                window.WorkbenchLoading.hide();
            }
            pendingThemeRerunConfirm = config;
            if (config.sourceBackdrop) {
                config.sourceBackdrop.classList.add("hidden");
            }
            themeRerunBackdrop.classList.remove("hidden");
            return true;
        }

        function closeManualFilterConfirm(restoreSource) {
            if (confirmBackdrop) {
                confirmBackdrop.classList.add("hidden");
            }
            if (
                restoreSource &&
                pendingManualFilterConfirm &&
                pendingManualFilterConfirm.sourceBackdrop
            ) {
                pendingManualFilterConfirm.sourceBackdrop.classList.remove("hidden");
            }
            pendingManualFilterConfirm = null;
        }

        function closeThemeRerunConfirm(restoreSource) {
            if (themeRerunBackdrop) {
                themeRerunBackdrop.classList.add("hidden");
            }
            if (
                restoreSource &&
                pendingThemeRerunConfirm &&
                pendingThemeRerunConfirm.sourceBackdrop
            ) {
                pendingThemeRerunConfirm.sourceBackdrop.classList.remove("hidden");
            }
            pendingThemeRerunConfirm = null;
        }

        if (confirmCancel && !confirmCancel.dataset.bound) {
            confirmCancel.dataset.bound = "1";
            confirmCancel.addEventListener("click", function () {
                closeManualFilterConfirm(true);
            });
        }
        if (confirmBackdrop && !confirmBackdrop.dataset.bound) {
            confirmBackdrop.dataset.bound = "1";
            confirmBackdrop.addEventListener("click", function (event) {
                if (event.target === confirmBackdrop) {
                    closeManualFilterConfirm(true);
                }
            });
        }
        if (confirmKeep && !confirmKeep.dataset.bound) {
            confirmKeep.dataset.bound = "1";
            confirmKeep.addEventListener("click", function () {
                if (!pendingManualFilterConfirm || !pendingManualFilterConfirm.form) {
                    return;
                }
                const pending = pendingManualFilterConfirm;
                pending.form.dataset.manualFilterDecision = "selection";
                closeManualFilterConfirm(false);
                pending.form.requestSubmit(pending.submitter || undefined);
            });
        }
        if (confirmDiscard && !confirmDiscard.dataset.bound) {
            confirmDiscard.dataset.bound = "1";
            confirmDiscard.addEventListener("click", function () {
                if (!pendingManualFilterConfirm || !pendingManualFilterConfirm.form) {
                    return;
                }
                const pending = pendingManualFilterConfirm;
                pending.form.dataset.manualFilterDecision = "discard";
                closeManualFilterConfirm(false);
                pending.form.requestSubmit(pending.submitter || undefined);
            });
        }
        if (themeRerunCancel && !themeRerunCancel.dataset.bound) {
            themeRerunCancel.dataset.bound = "1";
            themeRerunCancel.addEventListener("click", function () {
                closeThemeRerunConfirm(true);
            });
        }
        if (themeRerunBackdrop && !themeRerunBackdrop.dataset.bound) {
            themeRerunBackdrop.dataset.bound = "1";
            themeRerunBackdrop.addEventListener("click", function (event) {
                if (event.target === themeRerunBackdrop) {
                    closeThemeRerunConfirm(true);
                }
            });
        }
        if (themeRerunConfirm && !themeRerunConfirm.dataset.bound) {
            themeRerunConfirm.dataset.bound = "1";
            themeRerunConfirm.addEventListener("click", function () {
                if (!pendingThemeRerunConfirm || !pendingThemeRerunConfirm.form) {
                    return;
                }
                const pending = pendingThemeRerunConfirm;
                const confirmInput = getThemeRerunInput(pending.form);
                if (confirmInput) {
                    confirmInput.value = "1";
                }
                pending.form.dataset.themeRerunConfirmed = "1";
                closeThemeRerunConfirm(false);
                pending.form.requestSubmit(pending.submitter || undefined);
            });
        }

        function bindPopover(config) {
            const openButton = document.querySelector(config.openSelector);
            const cancelButton = document.querySelector(config.cancelSelector);
            const backdrop = byId(config.backdropId);
            const form = document.querySelector(config.formSelector);
            if (!openButton || !cancelButton || !backdrop || !form) {
                return;
            }

            const initialFocus = config.initialFocusId ? byId(config.initialFocusId) : null;

            function openPopover() {
                backdrop.classList.remove("hidden");
                const confirmInput = getThemeRerunInput(form);
                if (confirmInput) {
                    confirmInput.value = "0";
                }
                delete form.dataset.themeRerunConfirmed;
                if (initialFocus) {
                    initialFocus.focus();
                }
            }

            function closePopover(restoreFocus) {
                backdrop.classList.add("hidden");
                if (restoreFocus) {
                    openButton.focus();
                }
            }

            openButton.onclick = openPopover;
            cancelButton.onclick = function () {
                const confirmInput = getThemeRerunInput(form);
                if (confirmInput) {
                    confirmInput.value = "0";
                }
                delete form.dataset.themeRerunConfirmed;
                closePopover(true);
            };
            backdrop.onclick = function (event) {
                if (event.target === backdrop) {
                    const confirmInput = getThemeRerunInput(form);
                    if (confirmInput) {
                        confirmInput.value = "0";
                    }
                    delete form.dataset.themeRerunConfirmed;
                    closePopover(false);
                }
            };
            form.onsubmit = function (event) {
                if (typeof config.beforeSubmit === "function") {
                    const handled = config.beforeSubmit({
                        event: event,
                        form: form,
                        backdrop: backdrop,
                        submitter: event.submitter || null,
                    });
                    if (handled) {
                        return;
                    }
                }

                const decision = form.dataset.manualFilterDecision || "";
                const filters = getDashboardManualFilters();
                const hasFilters = hasDashboardManualFilters(filters);

                if (!decision && hasFilters) {
                    event.preventDefault();
                    const opened = openManualFilterConfirm({
                        actionLabel: config.actionLabel,
                        form: form,
                        submitter: event.submitter || null,
                        sourceBackdrop: backdrop,
                        filters: filters,
                    });
                    if (opened) {
                        return;
                    }
                }

                if (decision === "selection") {
                    setManualFilterHiddenInputs(form, filters);
                } else {
                    clearManualFilterHiddenInputs(form);
                }
                form.dataset.manualFilterDecision = "";
                closePopover(false);
            };
        }

        bindPopover({
            openSelector: "[data-advanced-ai-filter-open]",
            cancelSelector: "[data-advanced-ai-filter-cancel]",
            backdropId: "advanced-ai-popover-backdrop",
            formSelector: "#advanced-ai-popover-backdrop form",
            initialFocusId: "advanced_ai_search_query",
            loadingMessage: "Screening reports...",
            actionLabel: "filter reports",
        });

        bindPopover({
            openSelector: "[data-advanced-ai-discover-open]",
            cancelSelector: "[data-advanced-ai-discover-cancel]",
            backdropId: "advanced-ai-discover-popover",
            formSelector: "#advanced-ai-discover-popover form",
            initialFocusId: "extra_theme_instructions",
            loadingMessage: "Discovering themes and building a preview...",
            actionLabel: "discover themes",
            beforeSubmit: function (context) {
                const form = context.form;
                const confirmInput = getThemeRerunInput(form);
                if (!hasExistingThemes(form)) {
                    if (confirmInput) {
                        confirmInput.value = "0";
                    }
                    delete form.dataset.themeRerunConfirmed;
                    return false;
                }
                if (form.dataset.themeRerunConfirmed === "1") {
                    if (confirmInput) {
                        confirmInput.value = "1";
                    }
                    delete form.dataset.themeRerunConfirmed;
                    return false;
                }

                context.event.preventDefault();
                const opened = openThemeRerunConfirm({
                    form: form,
                    submitter: context.submitter || null,
                    sourceBackdrop: context.backdrop,
                });
                if (!opened) {
                    const approved = window.confirm(
                        "Theme discovery has already been applied. Run again and replace existing theme assignments?"
                    );
                    if (!approved) {
                        if (confirmInput) {
                            confirmInput.value = "0";
                        }
                        return true;
                    }
                    if (confirmInput) {
                        confirmInput.value = "1";
                    }
                    form.dataset.themeRerunConfirmed = "1";
                    form.requestSubmit(context.submitter || undefined);
                    return true;
                }
                return true;
            },
        });

        bindPopover({
            openSelector: "[data-advanced-ai-extract-open]",
            cancelSelector: "[data-advanced-ai-extract-cancel]",
            backdropId: "advanced-ai-extract-popover",
            formSelector: "#advanced-ai-extract-popover form",
            initialFocusId: "extract_extra_instructions",
            loadingMessage: "Extracting structured fields...",
            actionLabel: "extract data",
        });

        if (document.body.dataset.advancedAIEscapeBound === "1") {
            return;
        }
        document.body.dataset.advancedAIEscapeBound = "1";

        document.addEventListener("keydown", function (event) {
            if (event.key !== "Escape") {
                return;
            }
            if (pendingThemeRerunConfirm && themeRerunBackdrop && !themeRerunBackdrop.classList.contains("hidden")) {
                closeThemeRerunConfirm(true);
                return;
            }
            if (pendingManualFilterConfirm && confirmBackdrop && !confirmBackdrop.classList.contains("hidden")) {
                closeManualFilterConfirm(true);
                return;
            }
            const visibleBackdrops = document.querySelectorAll(
                ".advanced-ai-popover-backdrop:not(.hidden):not(.advanced-ai-popover-backdrop--locked)"
            );
            visibleBackdrops.forEach((node) => node.classList.add("hidden"));
        });
    }

    function setupThemeRerunConfirmModal() {
        const form = document.querySelector("form[data-theme-rerun-confirm-form]");
        const modal = byId("theme-rerun-confirm-modal");
        const cancelButton = document.querySelector("[data-theme-rerun-cancel]");
        const confirmButton = document.querySelector("[data-theme-rerun-confirm]");
        if (!form || !modal || modal.dataset.bound === "1") {
            return;
        }
        modal.dataset.bound = "1";

        const confirmInput = form.querySelector("input[data-confirm-rerun-themes]");
        let pendingSubmitter = null;

        function openModal(submitter) {
            pendingSubmitter = submitter || null;
            modal.classList.remove("hidden");
            document.body.classList.add("modal-open");
        }

        function closeModal(restoreFocus) {
            modal.classList.add("hidden");
            document.body.classList.remove("modal-open");
            if (restoreFocus && pendingSubmitter && typeof pendingSubmitter.focus === "function") {
                pendingSubmitter.focus();
            }
            pendingSubmitter = null;
        }

        form.addEventListener("submit", function (event) {
            const hasExisting = form.dataset.hasExistingThemes === "1";
            if (!hasExisting) {
                if (confirmInput) {
                    confirmInput.value = "0";
                }
                delete form.dataset.themeRerunConfirmed;
                return;
            }
            if (form.dataset.themeRerunConfirmed === "1") {
                if (confirmInput) {
                    confirmInput.value = "1";
                }
                delete form.dataset.themeRerunConfirmed;
                return;
            }
            event.preventDefault();
            if (confirmInput) {
                confirmInput.value = "0";
            }
            openModal(event.submitter || null);
        });

        if (cancelButton) {
            cancelButton.addEventListener("click", function () {
                closeModal(true);
            });
        }

        if (confirmButton) {
            confirmButton.addEventListener("click", function () {
                form.dataset.themeRerunConfirmed = "1";
                if (confirmInput) {
                    confirmInput.value = "1";
                }
                const submitter = pendingSubmitter;
                closeModal(false);
                form.requestSubmit(submitter || undefined);
            });
        }

        modal.addEventListener("click", function (event) {
            if (event.target === modal) {
                closeModal(true);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && !modal.classList.contains("hidden")) {
                closeModal(true);
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
            setupDatasetRowActions();
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

    function setupExcludedReportsCollapse() {
        const root = document.querySelector(".explore-surface--excluded[data-excluded-collapse-root]");
        if (!root) {
            return;
        }

        const toggle = root.querySelector("[data-excluded-collapse-toggle]");
        const label = root.querySelector("[data-excluded-collapse-label]");
        if (!toggle || !label) {
            return;
        }
        if (toggle.dataset.bound === "1") {
            return;
        }
        toggle.dataset.bound = "1";

        const storageKey = "workbench.excludedReportsExpanded";
        const excludedCount = Number(root.dataset.excludedCount || "0");
        let expanded = false;
        try {
            expanded = window.localStorage.getItem(storageKey) === "1";
        } catch (error) {
            expanded = false;
        }

        function paint(isExpanded) {
            root.classList.toggle("is-collapsed", !isExpanded);
            toggle.setAttribute("aria-expanded", isExpanded ? "true" : "false");
            const countText = Number.isFinite(excludedCount) ? ` (${excludedCount})` : "";
            label.textContent = isExpanded
                ? `Hide excluded reports${countText}`
                : `Show excluded reports${countText}`;
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

    function setupThemeSummaryCollapse() {
        const root = document.querySelector(".explore-surface--theme[data-theme-collapse-root]");
        if (!root) {
            return;
        }

        const toggle = root.querySelector("[data-theme-collapse-toggle]");
        const label = root.querySelector("[data-theme-collapse-label]");
        if (!toggle || !label) {
            return;
        }
        if (toggle.dataset.bound === "1") {
            return;
        }
        toggle.dataset.bound = "1";

        const storageKey = "workbench.themeSummaryExpanded";
        let expanded = false;
        try {
            expanded = window.localStorage.getItem(storageKey) === "1";
        } catch (error) {
            expanded = false;
        }

        function paint(isExpanded) {
            root.classList.toggle("is-collapsed", !isExpanded);
            toggle.setAttribute("aria-expanded", isExpanded ? "true" : "false");
            label.textContent = isExpanded ? "Hide thematic snapshot" : "Show thematic snapshot";
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

    function setupThemePickerAutoSave() {
        const picker = document.querySelector(".settings-theme-picker");
        if (!picker || picker.dataset.bound === "1") {
            return;
        }
        picker.dataset.bound = "1";

        const radios = picker.querySelectorAll("input[name='ui_theme']");
        if (!radios.length) {
            return;
        }

        radios.forEach((radio) => {
            radio.addEventListener("change", function () {
                if (!radio.checked) {
                    return;
                }

                const selectedTheme = String(radio.value || "").trim();
                if (!selectedTheme) {
                    return;
                }

                try {
                    window.localStorage.setItem("workbench.settingsThemeDetailsOpen", "1");
                } catch (error) {
                    // Ignore storage access issues.
                }

                // Apply immediately for instant feedback.
                document.body.setAttribute("data-ui-theme", selectedTheme);

                const csrfToken = getCsrfTokenFromPage();
                if (!csrfToken) {
                    return;
                }

                if (picker.dataset.submitting === "1") {
                    return;
                }
                picker.dataset.submitting = "1";

                // Deterministic persistence: submit a minimal POST immediately.
                const submitForm = document.createElement("form");
                submitForm.method = "post";
                submitForm.action = window.location.href;
                submitForm.className = "hidden";

                const csrfInput = document.createElement("input");
                csrfInput.type = "hidden";
                csrfInput.name = "csrfmiddlewaretoken";
                csrfInput.value = csrfToken;
                submitForm.appendChild(csrfInput);

                const actionInput = document.createElement("input");
                actionInput.type = "hidden";
                actionInput.name = "action";
                actionInput.value = "set_ui_theme";
                submitForm.appendChild(actionInput);

                const themeInput = document.createElement("input");
                themeInput.type = "hidden";
                themeInput.name = "ui_theme";
                themeInput.value = selectedTheme;
                submitForm.appendChild(themeInput);

                document.body.appendChild(submitForm);
                submitForm.submit();
            });
        });
    }

    function setupSettingsThemeDetailsPersistence() {
        const details = document.querySelector("[data-settings-theme-details]");
        if (!details || details.dataset.bound === "1") {
            return;
        }
        details.dataset.bound = "1";

        const storageKey = "workbench.settingsThemeDetailsOpen";
        let isOpen = false;
        try {
            isOpen = window.localStorage.getItem(storageKey) === "1";
        } catch (error) {
            isOpen = false;
        }
        if (isOpen) {
            details.open = true;
        }

        details.addEventListener("toggle", function () {
            try {
                window.localStorage.setItem(storageKey, details.open ? "1" : "0");
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
                const chartNodes = Object.values(chartRoots);
                const hasAllInstances = chartNodes.every((chartRoot) => {
                    const instance = window.echarts.getInstanceByDom(chartRoot);
                    if (!instance) {
                        return false;
                    }
                    if (typeof instance.getDom === "function" && instance.getDom() !== chartRoot) {
                        // Cached markup can carry stale instance IDs; clear mismatched instances.
                        if (typeof instance.dispose === "function") {
                            instance.dispose();
                        }
                        return false;
                    }
                    return true;
                });
                if (hasAllInstances) {
                    chartNodes.forEach((chartRoot) => {
                        const instance = window.echarts.getInstanceByDom(chartRoot);
                        if (instance) {
                            instance.resize();
                        }
                    });
                    return;
                }
            }
            delete root.dataset.bound;
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

        const options = payload && typeof payload.options === "object" ? payload.options : {};
        const selected = payload && typeof payload.selected === "object" ? payload.selected : {};
        let summary = payload && typeof payload.summary === "object" ? payload.summary : {};
        const coronerOptions = Array.isArray(options.coroners) ? options.coroners : [];
        const areaOptions = Array.isArray(options.areas) ? options.areas : [];
        const receiverOptions = Array.isArray(options.receivers) ? options.receivers : [];

        function chartPalette() {
            return {
                gridTextColor: getCssVar("--chart-axis-text", "rgba(209, 220, 255, 0.7)"),
                splitLineColor: getCssVar("--chart-axis-line", "rgba(156, 173, 232, 0.18)"),
                monthlyLine: getCssVar("--chart-monthly-line", "#66d7ff"),
                monthlyArea: getCssVar("--chart-monthly-area", "rgba(102, 215, 255, 0.18)"),
                coronerBar: getCssVar("--chart-coroner-bar", "#5b9dff"),
                areaBar: getCssVar("--chart-area-bar", "#7ac26b"),
                receiverBar: getCssVar("--chart-receiver-bar", "#f2a85a"),
                emptyTextColor: getCssVar("--chart-empty-text", "rgba(210, 220, 255, 0.7)"),
            };
        }

        function normaliseValue(value) {
            if (typeof value !== "string") {
                if (value === null || value === undefined) {
                    return "";
                }
                return String(value).trim();
            }
            return value.trim();
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

        function snapshotSelectedFilters() {
            return {
                coroner: Array.from(selectedFilters.coroner).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })),
                area: Array.from(selectedFilters.area).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })),
                receiver: Array.from(selectedFilters.receiver).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })),
            };
        }

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
            const panelBase = root.dataset.dashboardPanelBase || "/dataset-panel/";
            const browserBase = root.dataset.dashboardBrowserBase || "?page=";
            const panelUrl = `${panelBase}?page=1${query ? `&${query}` : ""}`;
            const targetUrl = `${browserBase}1${query ? `&${query}` : ""}`;
            window.WorkbenchDatasetPanel.fetchAndSwapDataset(panelUrl, targetUrl);
        }

        function renderMonthlyChart(seriesRows) {
            const palette = chartPalette();
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
                        axisLabel: { color: palette.gridTextColor, rotate: 30 },
                        axisLine: { lineStyle: { color: palette.splitLineColor } },
                    },
                    yAxis: {
                        type: "value",
                        axisLabel: { color: palette.gridTextColor },
                        splitLine: { lineStyle: { color: palette.splitLineColor } },
                    },
                    series: [
                        {
                            type: "line",
                            smooth: true,
                            showSymbol: false,
                            lineStyle: { width: 2.2, color: palette.monthlyLine },
                            itemStyle: { color: palette.monthlyLine },
                            areaStyle: { color: palette.monthlyArea },
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
                                    fill: palette.emptyTextColor,
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
            const palette = chartPalette();
            const labels = seriesRows.map((row) => row.name);
            const values = seriesRows.map((row) => row.value);
            chart.setOption(
                {
                    animationDuration: 240,
                    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
                    grid: { left: 140, right: 16, top: 24, bottom: 24 },
                    xAxis: {
                        type: "value",
                        axisLabel: { color: palette.gridTextColor },
                        splitLine: { lineStyle: { color: palette.splitLineColor } },
                    },
                    yAxis: {
                        type: "category",
                        inverse: true,
                        data: labels,
                        axisLabel: {
                            color: palette.gridTextColor,
                            overflow: "truncate",
                            width: 130,
                        },
                        axisLine: { lineStyle: { color: palette.splitLineColor } },
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
                                    fill: palette.emptyTextColor,
                                    fontSize: 13,
                                },
                            },
                        ]
                        : [],
                },
                true
            );
        }

        function renderDashboard(currentSummary) {
            const reportsShown = Number(currentSummary.reports_shown || 0);
            const reportsTotal = Number(currentSummary.reports_total || 0);
            const uniqueCoroners = Number(currentSummary.unique_coroners || 0);
            const receiverLinks = Number(currentSummary.receiver_links || 0);
            const monthlyRows = Array.isArray(currentSummary.monthly) ? currentSummary.monthly : [];
            const topCoroners = Array.isArray(currentSummary.top_coroners) ? currentSummary.top_coroners : [];
            const topAreas = Array.isArray(currentSummary.top_areas) ? currentSummary.top_areas : [];
            const topReceivers = Array.isArray(currentSummary.top_receivers) ? currentSummary.top_receivers : [];

            statReports.textContent = reportsShown.toLocaleString();
            statCoroners.textContent = uniqueCoroners.toLocaleString();
            statReceiverLinks.textContent = receiverLinks.toLocaleString();
            renderMonthlyChart(monthlyRows);
            renderBarChart(
                charts.coroner,
                topCoroners,
                chartPalette().coronerBar,
                "No coroner values for this filter."
            );
            renderBarChart(
                charts.area,
                topAreas,
                chartPalette().areaBar,
                "No area values for this filter."
            );
            renderBarChart(
                charts.receiver,
                topReceivers,
                chartPalette().receiverBar,
                "No receiver values for this filter."
            );

            setStatus(`Showing ${reportsShown.toLocaleString()} of ${reportsTotal.toLocaleString()} reports.`);
        }

        let summaryRequestId = 0;
        async function refreshDashboardSummary() {
            const requestId = ++summaryRequestId;
            const query = buildFilterQueryString();
            const dashboardBase = root.dataset.dashboardDataBase || "/dashboard-data/";
            const dashboardUrl = `${dashboardBase}${query ? `?${query}` : ""}`;
            setStatus("Loading dashboard...");

            let response;
            try {
                response = await fetch(dashboardUrl, {
                    credentials: "same-origin",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                    },
                });
            } catch (error) {
                if (requestId === summaryRequestId) {
                    setStatus("Dashboard could not be updated. Please retry.");
                }
                return;
            }
            if (!response.ok) {
                if (requestId === summaryRequestId) {
                    setStatus("Dashboard could not be updated. Please retry.");
                }
                return;
            }
            let responsePayload = {};
            try {
                responsePayload = await response.json();
            } catch (error) {
                responsePayload = {};
            }
            if (requestId !== summaryRequestId) {
                return;
            }
            summary = responsePayload && typeof responsePayload.summary === "object"
                ? responsePayload.summary
                : {};
            renderDashboard(summary);
        }

        function applyFilterSelectionChange() {
            syncBrowserUrl();
            refreshDatasetTable();
            refreshDashboardSummary();
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
                    renderFilterControl(fieldName, allOptions, searchInput, optionsNode, badgesNode);
                    applyFilterSelectionChange();
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
                    renderFilterControl(fieldName, allOptions, searchInput, optionsNode, badgesNode);
                    applyFilterSelectionChange();
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
            renderFilterControl("coroner", coronerOptions, searchCoroner, optionsCoroner, badgesCoroner);
            renderFilterControl("area", areaOptions, searchArea, optionsArea, badgesArea);
            renderFilterControl("receiver", receiverOptions, searchReceiver, optionsReceiver, badgesReceiver);
            applyFilterSelectionChange();
        }

        window.WorkbenchDashboardFilters = {
            getSelection: snapshotSelectedFilters,
            hasAny: function () {
                return Boolean(selectedFilters.coroner.size || selectedFilters.area.size || selectedFilters.receiver.size);
            },
            reset: resetFilters,
        };

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
        renderDashboard(summary);
    }

    function setupWorkbookControls() {
        const root = byId("workbook-controls");
        if (!root || root.dataset.bound === "1") {
            return;
        }
        root.dataset.bound = "1";

        const titleInput = byId("workbook-title-input");
        const shareButton = root.querySelector("[data-workbook-share]");
        const statusNode = byId("workbook-controls-status");
        if (!titleInput || !shareButton) {
            return;
        }

        function setStatus(text, isError) {
            if (!statusNode) {
                return;
            }
            statusNode.textContent = text || "";
            statusNode.style.color = isError
                ? getCssVar("--status-error-text", "rgba(255, 183, 183, 0.9)")
                : getCssVar("--status-muted-text", "rgba(205, 217, 255, 0.72)");
        }

        function getCsrfToken() {
            const cookie = document.cookie || "";
            const match = cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
            if (match && match[1]) {
                return decodeURIComponent(match[1]);
            }
            const csrfInput = document.querySelector("input[name='csrfmiddlewaretoken']");
            return csrfInput ? csrfInput.value : "";
        }

        function getFilters() {
            if (
                window.WorkbenchDashboardFilters &&
                typeof window.WorkbenchDashboardFilters.getSelection === "function"
            ) {
                return window.WorkbenchDashboardFilters.getSelection();
            }
            return { coroner: [], area: [], receiver: [] };
        }

        function getWorkbookState() {
            return {
                workbookId: (root.dataset.workbookId || "").trim(),
                editToken: (root.dataset.workbookEditToken || "").trim(),
                createUrl: (root.dataset.workbookCreateUrl || "").trim(),
                saveTemplate: (root.dataset.workbookSaveUrlTemplate || "").trim(),
            };
        }

        function setWorkbookState(payload) {
            if (!payload || typeof payload !== "object") {
                return;
            }
            if (payload.workbook_id) {
                root.dataset.workbookId = payload.workbook_id;
            }
            if (payload.edit_token) {
                root.dataset.workbookEditToken = payload.edit_token;
            }
            const currentUrl = new URL(window.location.href);
            if (payload.workbook_id) {
                currentUrl.searchParams.set("workbook", payload.workbook_id);
            }
            if (root.dataset.workbookEditToken) {
                currentUrl.searchParams.set("edit", root.dataset.workbookEditToken);
            }
            window.history.replaceState(window.history.state, "", currentUrl.toString());
        }

        function updateControlsEnabledState() {
            const hasTitle = Boolean((titleInput.value || "").trim());
            shareButton.disabled = !hasTitle;
        }

        function isValidWorksheetTitle(title) {
            return /^[A-Za-z0-9 -]+$/.test(title);
        }

        async function saveWorkbook() {
            const title = (titleInput.value || "").trim();
            if (!title) {
                setStatus("Name this workbook first.", true);
                updateControlsEnabledState();
                return null;
            }
            if (!isValidWorksheetTitle(title)) {
                setStatus("Use letters, numbers, spaces, and hyphens only.", true);
                updateControlsEnabledState();
                return null;
            }

            const state = getWorkbookState();
            if (!state.createUrl) {
                setStatus("Workbook endpoints are not configured.", true);
                return null;
            }

            const hasWorkbook = Boolean(state.workbookId && state.editToken);
            const targetUrl = hasWorkbook
                ? state.saveTemplate.replace(
                    "00000000-0000-0000-0000-000000000000",
                    state.workbookId
                )
                : state.createUrl;
            const body = {
                title,
                filters: getFilters(),
            };
            if (hasWorkbook) {
                body.edit_token = state.editToken;
            }

            setStatus("Saving...");
            shareButton.disabled = true;

            let response;
            try {
                response = await fetch(targetUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": getCsrfToken(),
                    },
                    credentials: "same-origin",
                    body: JSON.stringify(body),
                });
            } catch (error) {
                setStatus("Save failed. Check your connection.", true);
                updateControlsEnabledState();
                return null;
            }

            let payload = {};
            try {
                payload = await response.json();
            } catch (error) {
                payload = {};
            }

            if (!response.ok || !payload.ok) {
                setStatus(payload.error || "Save failed.", true);
                updateControlsEnabledState();
                return null;
            }

            setWorkbookState(payload);
            updateControlsEnabledState();
            setStatus("Worksheet saved.");
            return payload;
        }

        async function shareWorkbook() {
            const payload = await saveWorkbook();
            if (!payload || !payload.share_url) {
                return;
            }
            try {
                await navigator.clipboard.writeText(payload.share_url);
                setStatus("Share link copied.");
            } catch (error) {
                setStatus("Share link ready.");
                window.prompt("Copy this share link:", payload.share_url);
            }
        }

        titleInput.addEventListener("input", function () {
            setStatus("");
            updateControlsEnabledState();
        });
        shareButton.addEventListener("click", function () {
            shareWorkbook();
        });

        updateControlsEnabledState();
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

    function setupReadonlyCloneConfirm() {
        const modal = byId("readonly-clone-confirm");
        const openButton = document.querySelector("[data-readonly-clone-open]");
        if (!modal || !openButton || modal.dataset.bound === "1") {
            return;
        }
        modal.dataset.bound = "1";

        const cancelButton = modal.querySelector("[data-readonly-clone-cancel]");

        function openModal() {
            modal.classList.remove("hidden");
            document.body.classList.add("modal-open");
        }

        function closeModal() {
            modal.classList.add("hidden");
            document.body.classList.remove("modal-open");
        }

        openButton.addEventListener("click", openModal);

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

            if (
                (action === "filter_reports" || action === "discover_themes" || action === "extract_features") &&
                !form.dataset.manualFilterDecision &&
                window.WorkbenchDashboardFilters &&
                typeof window.WorkbenchDashboardFilters.hasAny === "function" &&
                window.WorkbenchDashboardFilters.hasAny()
            ) {
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
        setupExcludedReportsCollapse();
        setupThemeSummaryCollapse();
        setupSettingsThemeDetailsPersistence();
        setupThemePickerAutoSave();
        setupDatasetCellPreview();
        setupDatasetRowActions();
        setupDatasetMutations();
        setupRevealAnimations();
        setupStartOverConfirm();
        setupDownloadBundleModal();
        setupTopbarQuickSettings();
        setupTopbarLimitControls();
        setupTopbarLLMProviderToggle();
        setupTopbarAIResetConfirm();
        setupTopbarDateValidation();
        setupSettingsLoadReportsConfirm();
        setupAdvancedAIPopovers();
        setupThemeRerunConfirmModal();
        setupDatasetPagination();
        setupExploreDashboard();
        setupWorkbookControls();
        setupConfigModalDismiss();
        setupReadonlyCloneConfirm();

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

        function locationKeyFromWindow() {
            return `${window.location.pathname}${window.location.search || ""}`;
        }

        function getWorkspaceToken(rootNode) {
            if (!rootNode || !rootNode.dataset) {
                return "";
            }
            return rootNode.dataset.workspaceToken || "";
        }

        function snapshotPageHtmlForCache(rootNode) {
            if (!rootNode) {
                return "";
            }
            const clone = rootNode.cloneNode(true);
            const nodes = [clone, ...clone.querySelectorAll("*")];
            nodes.forEach((node) => {
                if (typeof node.getAttributeNames !== "function") {
                    return;
                }
                node.getAttributeNames().forEach((name) => {
                    const lower = String(name || "").toLowerCase();
                    if (lower === "data-bound" || (lower.startsWith("data-") && lower.endsWith("-bound"))) {
                        node.removeAttribute(name);
                    }
                });
            });
            return clone.innerHTML;
        }

        const pageCache = new Map();
        let activePath = locationKeyFromWindow();
        let activeWorkspaceToken = getWorkspaceToken(contentRoot);
        pageCache.set(activePath, {
            html: snapshotPageHtmlForCache(contentRoot),
            workspaceToken: activeWorkspaceToken,
        });

        function swapTo(pathname, page, html, pushHistory, workspaceToken) {
            if (typeof html !== "string") {
                return;
            }
            contentRoot.innerHTML = html;
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

            const targetPath = `${targetUrl.pathname}${targetUrl.search || ""}`;
            const currentPath = locationKeyFromWindow();
            if (targetUrl.pathname.startsWith("/for-coders") || window.location.pathname.startsWith("/for-coders")) {
                return;
            }
            if (targetPath === currentPath) {
                event.preventDefault();
                return;
            }

            event.preventDefault();
            activeWorkspaceToken = getWorkspaceToken(contentRoot);
            pageCache.set(currentPath, {
                html: snapshotPageHtmlForCache(contentRoot),
                workspaceToken: activeWorkspaceToken,
            });

            const targetPage = link.getAttribute("data-page-link") || getPageFromPath(targetPath);
            const cachedEntry = pageCache.get(targetPath);
            if (cachedEntry && cachedEntry.workspaceToken === activeWorkspaceToken) {
                swapTo(targetPath, targetPage, cachedEntry.html, true, cachedEntry.workspaceToken);
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
                    const incomingHtml = incomingRoot ? snapshotPageHtmlForCache(incomingRoot) : "";
                    const page = incomingRoot && incomingRoot.dataset.page
                        ? incomingRoot.dataset.page
                        : targetPage;
                    const workspaceToken = incomingRoot ? getWorkspaceToken(incomingRoot) : "";
                    if (!incomingRoot) {
                        window.location.href = targetPath;
                        return;
                    }
                    pageCache.set(targetPath, {
                        html: incomingHtml,
                        workspaceToken: workspaceToken,
                    });
                    swapTo(targetPath, page, incomingHtml, true, workspaceToken);
                })
                .catch(() => {
                    window.location.href = targetPath;
                });
        });

        window.addEventListener("popstate", function () {
            const path = locationKeyFromWindow();
            const page = getPageFromPath(window.location.pathname);
            activeWorkspaceToken = getWorkspaceToken(contentRoot);
            pageCache.set(activePath, {
                html: snapshotPageHtmlForCache(contentRoot),
                workspaceToken: activeWorkspaceToken,
            });
            const cachedEntry = pageCache.get(path);
            if (cachedEntry && cachedEntry.workspaceToken === activeWorkspaceToken) {
                swapTo(path, page, cachedEntry.html, false, cachedEntry.workspaceToken);
                return;
            }
            window.location.reload();
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        setupSidebarCollapse();
        setupForCodersLinkRouting();
        setupClientNavigation();
        initializePageFeatures();
    });
})();
