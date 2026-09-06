/* Small, dependency-free interactions for the paper's project page. */
(() => {
  'use strict';

  function initialize() {
    const one = (selector, parent = document) => parent.querySelector(selector);
    const all = (selector, parent = document) => [...parent.querySelectorAll(selector)];
    const byId = id => document.getElementById(id);

    const burger = one('.navbar-burger');
    const menu = byId('site-menu');
    function closeMenu() {
      burger?.classList.remove('is-active');
      burger?.setAttribute('aria-expanded', 'false');
      menu?.classList.remove('is-active');
    }
    burger?.addEventListener('click', () => {
      const open = burger.getAttribute('aria-expanded') !== 'true';
      burger.setAttribute('aria-expanded', String(open));
      burger.classList.toggle('is-active', open);
      menu?.classList.toggle('is-active', open);
    });
    menu?.addEventListener('click', event => {
      if (event.target.closest('a')) closeMenu();
    });
    document.addEventListener('click', event => {
      if (!burger?.contains(event.target) && !menu?.contains(event.target)) closeMenu();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && burger?.getAttribute('aria-expanded') === 'true') {
        closeMenu();
        burger.focus();
      }
    });

    const costTabs = ['cost-time', 'cost-memory', 'cost-energy'].map(byId).filter(Boolean);
    const costPanel = byId('cost-panel');
    const costImage = byId('cost-image');
    const costZoom = byId('cost-zoom');
    const costOriginal = byId('cost-original');
    const costCaption = byId('cost-caption');
    let currentTab = null;

    function selectCostTab(tab, moveFocus = false) {
      if (!tab || !tab.dataset.src) return;
      costTabs.forEach(candidate => {
        const active = candidate === tab;
        candidate.classList.toggle('is-active', active);
        candidate.setAttribute('aria-selected', String(active));
        candidate.tabIndex = active ? 0 : -1;
      });
      costPanel?.setAttribute('aria-labelledby', tab.id);
      // Assign the selected image and its metadata together. There is no
      // asynchronous callback that can restore an earlier tab after rapid input.
      if (costImage) {
        costImage.alt = tab.dataset.alt || '';
        if (currentTab !== tab) costImage.src = tab.dataset.src;
      }
      if (costZoom) {
        costZoom.href = tab.dataset.src;
        costZoom.dataset.original = tab.dataset.original || tab.dataset.src;
        costZoom.dataset.title = tab.dataset.alt || tab.textContent.trim();
      }
      if (costOriginal) costOriginal.href = tab.dataset.original || tab.dataset.src;
      if (costCaption) costCaption.textContent = tab.dataset.caption || '';
      currentTab = tab;
      if (moveFocus) tab.focus();
    }

    costTabs.forEach((tab, index) => {
      tab.addEventListener('click', () => selectCostTab(tab));
      tab.addEventListener('keydown', event => {
        const vertical = tab.closest('[role="tablist"]')?.getAttribute('aria-orientation') === 'vertical';
        let nextIndex;
        if (event.key === 'Home') nextIndex = 0;
        else if (event.key === 'End') nextIndex = costTabs.length - 1;
        else if (event.key === (vertical ? 'ArrowDown' : 'ArrowRight')) nextIndex = (index + 1) % costTabs.length;
        else if (event.key === (vertical ? 'ArrowUp' : 'ArrowLeft')) nextIndex = (index - 1 + costTabs.length) % costTabs.length;
        else return;
        event.preventDefault();
        selectCostTab(costTabs[nextIndex], true);
      });
    });
    selectCostTab(costTabs.find(tab => tab.getAttribute('aria-selected') === 'true') || costTabs[0]);

    // Warm only the three local figure assets used by these tabs.
    const preloadedSources = new Set();
    costTabs.forEach(tab => {
      const source = tab.dataset.src;
      if (!source || preloadedSources.has(source)) return;
      const resolved = new URL(source, document.baseURI);
      const local = resolved.origin === location.origin || (location.protocol === 'file:' && resolved.protocol === 'file:');
      if (!local) return;
      preloadedSources.add(source);
      const image = new Image();
      image.src = resolved.href;
    });

    const dialog = byId('figure-dialog');
    const dialogImage = byId('dialog-image');
    const dialogTitle = byId('dialog-title');
    const dialogOriginal = byId('dialog-original');
    let figureTrigger = null;
    if (dialog && typeof dialog.showModal === 'function') {
      document.addEventListener('click', event => {
        const link = event.target.closest('a.figure-zoom');
        if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        figureTrigger = link;
        const sourceImage = one('img', link);
        const title = link.dataset.title || sourceImage?.alt || 'Paper figure';
        if (dialogImage) {
          dialogImage.src = link.href;
          dialogImage.alt = sourceImage?.alt || title;
        }
        if (dialogTitle) dialogTitle.textContent = title;
        if (dialogOriginal) dialogOriginal.href = link.dataset.original || link.href;
        dialog.showModal();
        document.documentElement.classList.add('figure-dialog-open');
        one('.dialog-close', dialog)?.focus();
      });
      one('.dialog-close', dialog)?.addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', event => {
        if (event.target !== dialog) return;
        const rect = dialog.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
      });
      // Escape uses the native dialog cancel behavior; closing through any route
      // clears the scroll lock and returns focus to the original figure link.
      dialog.addEventListener('close', () => {
        document.documentElement.classList.remove('figure-dialog-open');
        figureTrigger?.focus({ preventScroll: true });
      });
    }

    const copyButton = byId('copy-bibtex');
    const citation = byId('bibtex-code');
    const copyStatus = byId('copy-status');
    let resetCopyLabel;
    copyButton?.addEventListener('click', async () => {
      if (!citation) return;
      const label = one('.copy-label', copyButton);
      let copied = false;
      if (window.isSecureContext && navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(citation.textContent.trim());
          copied = true;
        } catch { /* Try the selection-based browser fallback below. */ }
      }
      if (!copied) {
        const selection = window.getSelection();
        if (selection) {
          const range = document.createRange();
          range.selectNodeContents(citation);
          selection.removeAllRanges();
          selection.addRange(range);
          try { copied = document.execCommand('copy'); } catch { copied = false; }
          if (copied) selection.removeAllRanges();
        }
      }
      clearTimeout(resetCopyLabel);
      if (label) label.textContent = copied ? 'Copied!' : 'Citation selected';
      if (copyStatus) {
        copyStatus.textContent = copied ? 'BibTeX copied to clipboard.' : 'Citation selected. Press Ctrl+C (Windows/Linux) or Command+C (Mac) to copy.';
        if (!copied) {
          copyStatus.hidden = false;
          copyStatus.classList.remove('sr-only', 'is-sr-only');
          citation.closest('pre')?.scrollIntoView({ block: 'nearest', behavior: 'auto' });
        }
      }
      resetCopyLabel = setTimeout(() => {
        if (label) label.textContent = 'Copy BibTeX';
      }, 3000);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
})();
