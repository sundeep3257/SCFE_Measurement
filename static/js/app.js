(function () {
  // Results page tabs (must run even when upload form is absent)
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  if (tabs.length && panels.length) {
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        tabs.forEach((t) => {
          const isActive = t === tab;
          t.classList.toggle("active", isActive);
          t.setAttribute("aria-selected", isActive ? "true" : "false");
        });
        panels.forEach((panel) => {
          const isActive = panel.id === `panel-${target}`;
          panel.classList.toggle("active", isActive);
          panel.hidden = !isActive;
        });
      });
    });
  }

  // Upload page logic
  const form = document.getElementById("upload-form");
  if (!form) return;

  const fileInput = document.getElementById("image");
  const dropzone = document.getElementById("dropzone");
  const previewArea = document.getElementById("preview-area");
  const previewImage = document.getElementById("preview-image");
  const previewFilename = document.getElementById("preview-filename");
  const analyzeBtn = document.getElementById("analyze-btn");
  const overlay = document.getElementById("processing-overlay");

  function setFile(file) {
    if (!file) return;
    if (!file.type.includes("png")) {
      alert("Please select a PNG image.");
      return;
    }
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    previewFilename.textContent = file.name;
    previewImage.src = URL.createObjectURL(file);
    previewArea.classList.remove("hidden");
    analyzeBtn.disabled = false;
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#2c5282";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    setFile(file);
  });

  form.addEventListener("submit", () => {
    analyzeBtn.disabled = true;
    overlay.classList.remove("hidden");
  });
})();
