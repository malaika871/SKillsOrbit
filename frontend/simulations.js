/** Shared live simulation animation helpers for SkillOrbit */
const SimUtils = {
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),

    async animatePetriFromLog(log, opts = {}) {
        const {
            placePrefix = 'place-',
            transPrefix = 'trans-',
            tokenId = 'token',
            msgId = 'petri-current-msg',
            placeCoords = {},
            stepDelay = 700,
        } = opts;

        const token = document.getElementById(tokenId);
        const msgEl = msgId ? document.getElementById(msgId) : null;

        for (const entry of log) {
            if (msgEl && entry.description) msgEl.textContent = entry.description;

            const place = document.getElementById(placePrefix + entry.active_place);
            if (place) {
                document.querySelectorAll('[id^="' + placePrefix + '"]').forEach((p) => {
                    p.classList.remove('active');
                });
                place.classList.add('active');
                place.classList.add('done');
            }

            if (entry.fired_transition) {
                const trans = document.getElementById(transPrefix + entry.fired_transition);
                if (trans) {
                    trans.classList.add('active');
                    trans.classList.add('done');
                }
            }

            if (token && placeCoords[entry.active_place] != null) {
                token.setAttribute('cx', placeCoords[entry.active_place]);
            }

            await this.sleep(stepDelay);
        }
    },

    async animatePipeline(steps, opts = {}) {
        const {
            msgId = 'pipeline-msg',
            barId = 'pipeline-bar',
            chipsId = 'pipeline-chips',
            delay = 650,
        } = opts;

        const msgEl = document.getElementById(msgId);
        const barEl = document.getElementById(barId);
        const chipsEl = document.getElementById(chipsId);

        for (const step of steps) {
            if (msgEl) msgEl.textContent = step.description;
            if (barEl && step.match_percentage != null) {
                barEl.style.width = step.match_percentage + '%';
            }
            if (chipsEl && step.skill) {
                const chip = Array.from(chipsEl.querySelectorAll('[data-skill]')).find(
                    (el) => el.getAttribute('data-skill') === step.skill
                );
                if (chip) chip.classList.replace('pending', 'lit');
            }
            await this.sleep(delay);
        }
    },

    async animateJourneySteps(steps, opts = {}) {
        const { containerId = 'journey-steps', delay = 800 } = opts;
        const container = document.getElementById(containerId);
        if (!container) return;

        const nodes = container.querySelectorAll('.journey-step');
        for (let i = 0; i < steps.length; i++) {
            nodes.forEach((n, j) => {
                n.classList.remove('active');
                if (j < i) n.classList.add('done');
            });
            if (nodes[i]) {
                nodes[i].classList.add('active');
                nodes[i].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            await this.sleep(delay);
        }
        if (nodes.length) {
            nodes[nodes.length - 1].classList.remove('active');
            nodes[nodes.length - 1].classList.add('done');
        }
    },

    async animateMarkovPath(pathLog, opts = {}) {
        const { containerId = 'markov-states', delay = 700 } = opts;
        const container = document.getElementById(containerId);
        if (!container) return;

        const nodes = container.querySelectorAll('.markov-node');
        for (const entry of pathLog) {
            nodes.forEach((n) => n.classList.remove('active'));
            const idx = entry.state_index != null ? entry.state_index : entry.step;
            if (nodes[idx]) {
                nodes[idx].classList.add('active');
                nodes.forEach((n, j) => { if (j < idx) n.classList.add('done'); });
            }
            const msg = document.getElementById('markov-msg');
            if (msg) msg.textContent = entry.description;
            await this.sleep(delay);
        }
    },

    async animateQueue(events, opts = {}) {
        const { barId = 'queue-bar', msgId = 'queue-msg', delay = 450 } = opts;
        const barWrap = document.getElementById(barId);
        const msgEl = document.getElementById(msgId);
        const maxQ = Math.max(...events.map((e) => e.queue_length || 0), 1);

        for (const ev of events) {
            if (msgEl) msgEl.textContent = ev.description;
            if (barWrap) {
                const h = Math.max(8, ((ev.queue_length || 0) / maxQ) * 100);
                barWrap.style.height = h + '%';
            }
            await this.sleep(delay);
        }
    },

    buildRoadmapPetriSvg(phases, svgId, width = 800) {
        const n = phases.length;
        if (n === 0) return;
        const svg = document.getElementById(svgId);
        if (!svg) return;

        const spacing = (width - 120) / Math.max(n - 1, 1);
        const coords = {};
        let html = '<defs><marker id="arrow-sim" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="rgba(195,192,255,0.45)"/></marker></defs>';

        phases.forEach((ph, i) => {
            const cx = 60 + i * spacing;
            const pid = 'P' + i;
            coords[pid] = cx;
            if (i > 0) {
                const x1 = coords['P' + (i - 1)] + 26;
                const x2 = cx - 26;
                html += `<line class="pn-arrow" x1="${x1}" y1="60" x2="${x2}" y2="60" marker-end="url(#arrow-sim)"/>`;
            }
            html += `<circle class="pn-place" id="place-${pid}" cx="${cx}" cy="60" r="24"/>`;
            html += `<text class="pn-lbl" x="${cx}" y="60">${pid}</text>`;
            const label = (ph.phase || '').replace(/^Phase \d+:\s*/, '').slice(0, 14);
            html += `<text class="pn-lbl-sm" x="${cx}" y="98">${label}</text>`;
            if (i < n - 1) {
                const tx = cx + 30;
                html += `<rect class="pn-trans" id="trans-T${i}" x="${tx}" y="48" width="36" height="24" rx="4"/>`;
                html += `<text class="pn-lbl" x="${tx + 18}" y="60">T${i}</text>`;
            }
        });

        html += `<circle class="pn-token-dot" id="token" cx="${coords.P0 || 60}" cy="60" r="7"/>`;
        svg.innerHTML = html;
        return coords;
    },
};
