/**
 * Multimodal Attendance & Engagement Agent - Reactive Dashboard Controller
 * Handles real-time polling, Chart.js timeline rendering, audio visualizer,
 * and live attendance updates.
 */

let timelineChart = null;
let isSimulatorMode = true;

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    startPolling();
    initAudioBars();
    bindEvents();
});

// ================= Chart Initialization =================
function initChart() {
    const ctx = document.getElementById("engagementTimelineChart");
    if (!ctx) return;

    timelineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Classroom Engagement (%)",
                    data: [],
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.12)",
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointRadius: (context) => {
                        const raw = context.raw;
                        return (raw && (raw.drop || raw.peak)) ? 6 : 2;
                    },
                    pointBackgroundColor: (context) => {
                        const raw = context.raw;
                        if (!raw) return "#38bdf8";
                        if (raw.drop) return "#ef4444";
                        if (raw.peak) return "#10b981";
                        return "#38bdf8";
                    },
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 1.5,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8", font: { size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#64748b", font: { size: 10 }, maxTicksLimit: 12 }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#111827",
                    titleColor: "#f8fafc",
                    bodyColor: "#94a3b8",
                    borderColor: "rgba(56, 189, 248, 0.3)",
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: (ctx) => {
                            const raw = ctx.raw;
                            let msg = ` Engagement: ${raw.y}%`;
                            if (raw.drop) msg += " [Attention Dip]";
                            if (raw.peak) msg += " [Interaction Peak]";
                            return msg;
                        }
                    }
                }
            }
        }
    });
}

// ================= State Polling & DOM Updates =================
function startPolling() {
    setInterval(fetchState, 1000);
    fetchState();
}

async function fetchState() {
    try {
        const res = await fetch("/api/state");
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "initializing") return;

        updateKPIs(data);
        updateDistribution(data);
        updateAudio(data.audio);
        updateSuggestions(data.suggestions);
        updateAttendanceTable(data.attendance);
        updateTimeline(data.timeline);

        const fpsEl = document.getElementById("fpsIndicator");
        if (fpsEl) fpsEl.textContent = `${data.fps} FPS`;

    } catch (e) {
        console.warn("Polling error:", e);
    }
}

function updateKPIs(data) {
    const engVal = document.getElementById("kpiEngagement");
    if (engVal) {
        engVal.textContent = Math.round(data.overall_engagement);
        const score = data.overall_engagement;
        if (score >= 75) {
            engVal.style.color = "#34d399";
        } else if (score >= 50) {
            engVal.style.color = "#38bdf8";
        } else {
            engVal.style.color = "#f43f5e";
        }
    }

    const trendEl = document.getElementById("kpiTrend");
    if (trendEl) {
        trendEl.textContent = data.trend;
        trendEl.className = "kpi-badge " + (
            data.trend === "RISING" ? "kpi-badge-positive" :
            data.trend === "FALLING" ? "kpi-badge-danger" : "kpi-badge-neutral"
        );
    }

    const activeEl = document.getElementById("kpiActiveStudents");
    if (activeEl) activeEl.textContent = data.active_students;

    const attRateEl = document.getElementById("kpiAttendanceRate");
    if (attRateEl && data.attendance) {
        attRateEl.textContent = `${data.attendance.attendance_rate}%`;
    }

    const verifiedEl = document.getElementById("kpiVerifiedPresent");
    if (verifiedEl && data.attendance) {
        verifiedEl.textContent = data.attendance.present_count;
    }
}

function updateDistribution(data) {
    const pAtt = document.getElementById("progAttentive");
    const pNotes = document.getElementById("progNotes");
    const pDist = document.getElementById("progDistracted");
    const pDrowsy = document.getElementById("progDrowsy");

    if (pAtt) pAtt.style.width = `${data.attentive_pct}%`;
    if (pNotes) pNotes.style.width = `${data.notes_pct}%`;
    if (pDist) pDist.style.width = `${data.distracted_pct}%`;
    if (pDrowsy) pDrowsy.style.width = `${data.drowsy_pct}%`;

    const lAtt = document.getElementById("legendAttentive");
    const lNotes = document.getElementById("legendNotes");
    const lDist = document.getElementById("legendDistracted");
    const lDrowsy = document.getElementById("legendDrowsy");

    if (lAtt) lAtt.textContent = `Attentive: ${data.attentive_pct}%`;
    if (lNotes) lNotes.textContent = `Notes: ${data.notes_pct}%`;
    if (lDist) lDist.textContent = `Distracted: ${data.distracted_pct}%`;
    if (lDrowsy) lDrowsy.textContent = `Drowsy: ${data.drowsy_pct}%`;
}

function updateAudio(audio) {
    if (!audio) return;
    const envChip = document.getElementById("audioEnvChip");
    if (envChip) {
        envChip.textContent = audio.environment.replace(/_/g, " ");
        if (audio.environment === "CLASSROOM_DISCUSSION") {
            envChip.style.borderColor = "#34d399";
            envChip.style.color = "#34d399";
        } else if (audio.environment === "SIDE_CHATTER") {
            envChip.style.borderColor = "#fbbf24";
            envChip.style.color = "#fbbf24";
        } else {
            envChip.style.borderColor = "rgba(56, 189, 248, 0.3)";
            envChip.style.color = "#38bdf8";
        }
    }

    const vadEl = document.getElementById("vadStatus");
    if (vadEl) {
        vadEl.textContent = audio.vad_active ? "Speech Active" : "Silence / Ambient";
        vadEl.className = "kpi-badge " + (audio.vad_active ? "kpi-badge-positive" : "kpi-badge-neutral");
    }

    // Animate audio wave bars based on RMS
    const bars = document.querySelectorAll(".audio-bar");
    const baseHeight = Math.min(26, Math.max(4, audio.rms * 320));
    bars.forEach((bar, i) => {
        const jitter = Math.sin(Date.now() * 0.01 + i) * 6;
        bar.style.height = `${Math.max(4, baseHeight + jitter)}px`;
    });
}

function updateSuggestions(suggestions) {
    const feed = document.getElementById("suggestionsFeed");
    if (!feed) return;

    if (!suggestions || suggestions.length === 0) {
        feed.innerHTML = `
            <div style="text-align:center; padding: 2.5rem; color: #64748b; font-size: 0.85rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">✨</div>
                All classroom metrics within optimal range.<br>
                Proactive teaching interventions will appear here as session dynamics evolve.
            </div>`;
        return;
    }

    feed.innerHTML = suggestions.map(s => {
        const cardClass = s.urgency === "HIGH" ? "card-high" : (s.urgency === "MEDIUM" ? "card-medium" : "card-info");
        const badgeClass = s.urgency === "HIGH" ? "kpi-badge-danger" : (s.urgency === "MEDIUM" ? "kpi-badge-warning" : "kpi-badge-positive");

        const evidenceChips = Object.entries(s.evidence || {}).map(([k, v]) => `
            <span class="evidence-chip"><strong>${k.replace(/_/g, ' ')}:</strong> ${v}</span>
        `).join("");

        return `
            <div class="suggestion-card ${cardClass}">
                <div class="sug-header">
                    <span class="sug-title">${escapeHtml(s.title)}</span>
                    <span class="kpi-badge ${badgeClass}">${s.urgency}</span>
                </div>
                <p class="sug-prompt">${escapeHtml(s.action_prompt)}</p>
                <div class="sug-rationale">
                    <strong>Pedagogical Rationale:</strong> ${escapeHtml(s.rationale)}
                </div>
                <div class="sug-evidence-chips">
                    ${evidenceChips}
                </div>
            </div>`;
    }).join("");
}

function updateAttendanceTable(attendance) {
    const tbody = document.getElementById("attendanceTableBody");
    if (!tbody || !attendance || !attendance.records) return;

    tbody.innerHTML = attendance.records.map(r => {
        let badgeClass = "status-absent";
        if (r.status === "PRESENT") badgeClass = "status-present";
        else if (r.status === "LATE") badgeClass = "status-late";
        else if (r.status === "PARTIAL") badgeClass = "status-partial";

        const verifiedBadge = r.sustained_verified
            ? `<span style="color:#34d399; font-weight:700;">✓ Sustained</span>`
            : `<span style="color:#94a3b8;">Pending (${r.active_duration_min}m)</span>`;

        return `
            <tr>
                <td><strong>${escapeHtml(r.student_id)}</strong></td>
                <td>${escapeHtml(r.student_name)}</td>
                <td>${escapeHtml(r.roll_no)}</td>
                <td>${escapeHtml(r.department)}</td>
                <td>${escapeHtml(r.first_seen)}</td>
                <td>${verifiedBadge}</td>
                <td><strong>${r.avg_engagement}%</strong></td>
                <td><span class="status-badge ${badgeClass}">${r.status}</span></td>
            </tr>`;
    }).join("");
}

function updateTimeline(timeline) {
    if (!timelineChart || !timeline || timeline.length === 0) return;

    const labels = timeline.map(t => t.time);
    const dataPoints = timeline.map(t => ({
        x: t.time,
        y: t.score,
        drop: t.drop,
        peak: t.peak
    }));

    timelineChart.data.labels = labels;
    timelineChart.data.datasets[0].data = dataPoints;
    timelineChart.update("none");
}

function initAudioBars() {
    const container = document.getElementById("audioBarsContainer");
    if (!container) return;
    container.innerHTML = "";
    for (let i = 0; i < 14; i++) {
        const bar = document.createElement("div");
        bar.className = "audio-bar";
        container.appendChild(bar);
    }
}

// ================= Event Handlers =================
function bindEvents() {
    // Mode Switcher Toggle
    const modeBtn = document.getElementById("btnToggleMode");
    if (modeBtn) {
        modeBtn.addEventListener("click", async () => {
            isSimulatorMode = !isSimulatorMode;
            modeBtn.textContent = isSimulatorMode ? "Switch to Live Camera" : "Switch to Classroom Simulator";
            try {
                await fetch("/api/mode", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ use_simulator: isSimulatorMode })
                });
            } catch (e) {
                console.error("Mode toggle error:", e);
            }
        });
    }

    // Export CSV
    const exportBtn = document.getElementById("btnExportAttendance");
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            window.location.href = "/api/export_attendance";
        });
    }

    // Reset Session
    const resetBtn = document.getElementById("btnResetSession");
    if (resetBtn) {
        resetBtn.addEventListener("click", async () => {
            if (confirm("Reset current attendance records and start a fresh session?")) {
                await fetch("/api/reset_session", { method: "POST" });
                fetchState();
            }
        });
    }

    // Modal open/close
    const openModalBtn = document.getElementById("btnEnrollModal");
    const closeModalBtn = document.getElementById("btnCloseModal");
    const modalOverlay = document.getElementById("enrollModal");

    if (openModalBtn && modalOverlay) {
        openModalBtn.addEventListener("click", () => modalOverlay.classList.add("open"));
    }
    if (closeModalBtn && modalOverlay) {
        closeModalBtn.addEventListener("click", () => modalOverlay.classList.remove("open"));
    }

    // Student Enrollment Form
    const enrollForm = document.getElementById("enrollStudentForm");
    if (enrollForm) {
        enrollForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const sId = document.getElementById("enrollId").value;
            const sName = document.getElementById("enrollName").value;
            const sRoll = document.getElementById("enrollRoll").value;
            const sDept = document.getElementById("enrollDept").value;
            const photoInput = document.getElementById("enrollPhoto");

            let photoB64 = "";
            if (photoInput && photoInput.files && photoInput.files[0]) {
                photoB64 = await toBase64(photoInput.files[0]);
            }

            try {
                const res = await fetch("/api/enroll", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        student_id: sId,
                        name: sName,
                        roll_no: sRoll,
                        department: sDept,
                        photo_base64: photoB64
                    })
                });
                const result = await res.json();
                if (result.success) {
                    alert(`Student ${sName} successfully enrolled!`);
                    modalOverlay.classList.remove("open");
                    enrollForm.reset();
                    fetchState();
                } else {
                    alert("Enrollment failed: " + (result.error || "Unknown"));
                }
            } catch (err) {
                alert("Error submitting enrollment: " + err);
            }
        });
    }
}

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

function escapeHtml(text) {
    if (!text) return "";
    return text.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
