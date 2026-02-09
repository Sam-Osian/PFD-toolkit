(function () {
    function byId(id) {
        return document.getElementById(id);
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

    document.addEventListener("DOMContentLoaded", function () {
        toggleProviderFields();
        toggleDiscoverTrimFields();
        toggleTruncationTypeFields();
        setupFeatureGrid();

        const provider = byId("provider_override");
        if (provider) {
            provider.addEventListener("change", toggleProviderFields);
        }

        const trimApproach = byId("trim_approach");
        if (trimApproach) {
            trimApproach.addEventListener("change", toggleDiscoverTrimFields);
        }

        const truncType = byId("truncation_limit_type");
        if (truncType) {
            truncType.addEventListener("change", toggleTruncationTypeFields);
        }
    });
})();
