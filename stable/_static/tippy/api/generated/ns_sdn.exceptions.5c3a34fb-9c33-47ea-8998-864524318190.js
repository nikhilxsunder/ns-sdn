selector_to_html = {"a[href=\"#exceptions\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Exceptions<a class=\"headerlink\" href=\"#exceptions\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#module-ns_sdn.exceptions\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">ns_sdn.exceptions<a class=\"headerlink\" href=\"#module-ns_sdn.exceptions\" title=\"Link to this heading\">#</a></h1><p>The error hierarchy every module raises from.</p><p>Three subclasses, one per kind of fix. <code class=\"docutils literal notranslate\"><span class=\"pre\">DimensionError</span></code> means the\ngeometry of an input is wrong (reshape the data); <code class=\"docutils literal notranslate\"><span class=\"pre\">SpecificationError</span></code>\nmeans the content of a choice is wrong (change the argument);\n<code class=\"docutils literal notranslate\"><span class=\"pre\">NumericalError</span></code> means the arithmetic could not be carried out (look at\nthe numbers). Each also derives from the matching builtin, so code written\nagainst <code class=\"docutils literal notranslate\"><span class=\"pre\">ValueError</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">RuntimeError</span></code> keeps working. The shape\nmirrors <code class=\"docutils literal notranslate\"><span class=\"pre\">cultivars.exceptions</span></code> so a caller handling one package\u2019s errors\nhandles the other\u2019s without a second <code class=\"docutils literal notranslate\"><span class=\"pre\">except</span></code> clause.</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(`article.bd-article ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }
            link.classList.add('has-tippy');
            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: true, theme: 'material', delay: [200, 0],
                onShow(instance) {MathJax.typesetPromise([instance.popper]).then(() => {});},
            });
        };
    };
    console.log("tippy tips loaded!");
};
