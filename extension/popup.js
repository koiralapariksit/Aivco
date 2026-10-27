document.addEventListener("DOMContentLoaded", () => {
  const statusBadge = document.getElementById("statusBadge");

  function updateUI() {
    chrome.runtime.sendMessage({ action: "get_status" }, (response) => {
      if (chrome.runtime.lastError || !response) {
        statusBadge.className = "badge badge-disconnected";
        statusBadge.textContent = "● Disconnected";
        return;
      }

      if (response.connected) {
        statusBadge.className = "badge badge-connected";
        statusBadge.textContent = "● Connected";
      } else {
        statusBadge.className = "badge badge-disconnected";
        statusBadge.textContent = "● Disconnected";
      }

      const tabs = response.tabs || {};
      for (const [provider, isOpen] of Object.entries(tabs)) {
        const dot = document.getElementById(`dot-${provider}`);
        const label = document.getElementById(`label-${provider}`);
        if (dot && label) {
          if (isOpen) {
            dot.className = "dot dot-ready";
            label.textContent = "Active Tab";
            label.style.color = "#10b981";
          } else {
            dot.className = "dot dot-offline";
            label.textContent = "Not Open";
            label.style.color = "#64748b";
          }
        }
      }
    });
  }

  updateUI();
  setInterval(updateUI, 2000);
});

