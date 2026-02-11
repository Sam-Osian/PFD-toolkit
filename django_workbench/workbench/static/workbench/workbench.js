(function () {
    document.documentElement.classList.add("js-enabled");
    const PAGE_CLASS_PREFIX = "page-";
    const KNOWN_PAGES = ["home", "explore", "filter", "themes", "extract", "settings"];

    function byId(id) {
        return document.getElementById(id);
    }

    function getPageFromPath(pathname) {
        if (pathname.startsWith("/explore-pfds")) {
            return "explore";
        }
        if (pathname.startsWith("/filter")) {
            return "filter";
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

    function setupDatasetPagination() {
        if (document.body.dataset.paginationBound === "1") {
            return;
        }
        document.body.dataset.paginationBound = "1";

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
            const panelUrl = `${panelBase}?page=${page}`;
            const targetUrl = `${browserBase}${page}`;
            fetchAndSwapDataset(panelUrl, targetUrl);
        });
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

    function initializePageFeatures() {
        setupTableScrollbars();
        setupPageScrollbar();
        toggleProviderFields();
        toggleDiscoverTrimFields();
        toggleTruncationTypeFields();
        setupFeatureGrid();
        setupDatasetCellPreview();
        setupRevealAnimations();
        setupStartOverConfirm();
        setupDownloadBundleModal();
        setupDatasetPagination();
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

        const pageCache = new Map();
        let activePath = window.location.pathname;
        if (contentRoot.firstElementChild) {
            pageCache.set(activePath, contentRoot.firstElementChild);
        }

        function swapTo(pathname, page, node, pushHistory) {
            if (!node) {
                return;
            }
            contentRoot.replaceChildren(node);
            contentRoot.dataset.page = page;
            applyBodyPageClass(page);
            syncSidebarActive(page);
            if (pushHistory) {
                window.history.pushState({ path: pathname, page: page }, "", pathname);
            }
            activePath = pathname;
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
                pageCache.set(currentPath, currentNode);
            }

            const targetPage = link.getAttribute("data-page-link") || getPageFromPath(targetPath);
            if (pageCache.has(targetPath)) {
                swapTo(targetPath, targetPage, pageCache.get(targetPath), true);
                return;
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
                    if (!incomingNode) {
                        window.location.href = targetPath;
                        return;
                    }
                    pageCache.set(targetPath, incomingNode);
                    swapTo(targetPath, page, incomingNode, true);
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
                pageCache.set(activePath, currentNode);
            }
            if (pageCache.has(path)) {
                swapTo(path, page, pageCache.get(path), false);
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
