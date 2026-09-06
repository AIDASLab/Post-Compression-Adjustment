/* Conceptual parameter scope with recovery gains transcribed from original Figure 4. */
(() => {
  'use strict';

  const scopes = {
    router: {
      ftGain: 0.23,
      kdGain: 0.17,
      title: 'Router only',
      description: 'Only router parameters are updated. Retained experts and shared model components remain frozen.',
      expertState: 'All retained experts frozen',
      expertLabel: 'All retained experts are frozen; expert icons are symbolic.',
      symbolicTrainable: 0,
      sharedTrainable: false
    },
    top8: {
      ftGain: 1.21,
      kdGain: 0.85,
      title: 'Router + top-8 experts',
      description: 'The router and the top-8 selected experts in each layer are updated. Unselected experts and shared model components remain frozen.',
      expertState: 'Top-8 selected experts per layer trainable',
      expertLabel: 'The top-8 selected experts per layer are trainable; the highlighted icons do not represent an exact count or proportion.',
      symbolicTrainable: 2,
      sharedTrainable: false
    },
    top16: {
      ftGain: 1.35,
      kdGain: 0.92,
      title: 'Router + top-16 experts',
      description: 'The router and the top-16 selected experts in each layer are updated. Unselected experts and shared model components remain frozen.',
      expertState: 'Top-16 selected experts per layer trainable',
      expertLabel: 'The top-16 selected experts per layer are trainable; the highlighted icons do not represent an exact count or proportion.',
      symbolicTrainable: 4,
      sharedTrainable: false
    },
    top50: {
      ftGain: 1.97,
      kdGain: 1.35,
      title: 'Router + top-50 experts',
      description: 'The router and the top-50 selected experts in each layer are updated. Unselected experts and shared model components remain frozen.',
      expertState: 'Top-50 selected experts per layer trainable',
      expertLabel: 'The top-50 selected experts per layer are trainable; the highlighted icons do not represent an exact count or proportion.',
      symbolicTrainable: 6,
      sharedTrainable: false
    },
    experts: {
      ftGain: 2.31,
      kdGain: 1.51,
      title: 'Router + all experts',
      description: 'The router and every retained expert are updated. Attention and other shared model parameters remain frozen.',
      expertState: 'All retained experts trainable',
      expertLabel: 'All retained experts are trainable; expert icons are symbolic.',
      symbolicTrainable: 8,
      sharedTrainable: false
    },
    full: {
      ftGain: 3.49,
      kdGain: 2.19,
      title: 'Full parameter',
      description: 'All model parameters are updated, including the router, retained experts, attention, and other shared components.',
      expertState: 'All retained experts trainable',
      expertLabel: 'All retained experts are trainable; expert icons are symbolic.',
      symbolicTrainable: 8,
      sharedTrainable: true
    }
  };

  function initialize() {
    document.querySelectorAll('[data-scope-explorer]').forEach(explorer => {
      const choices = [...explorer.querySelectorAll('.scope-choice[data-scope]')];
      const schematic = explorer.querySelector('.scope-schematic');
      const sharedModules = [...explorer.querySelectorAll('.scope-shared')];
      const expertIcons = [...explorer.querySelectorAll('.scope-expert')];
      const experts = explorer.querySelector('.scope-experts');
      const expertStatus = explorer.querySelector('.scope-experts-status');
      const title = explorer.querySelector('.scope-description-title');
      const description = explorer.querySelector('.scope-description');
      const selectedScope = explorer.querySelector('.scope-performance-selection');
      const ftGain = explorer.querySelector('#scope-ft');
      const kdGain = explorer.querySelector('#scope-kd');
      const ftBar = explorer.querySelector('#scope-ft-bar');
      const kdBar = explorer.querySelector('#scope-kd-bar');

      function selectScope(name) {
        const scope = scopes[name];
        if (!scope) return;
        explorer.dataset.activeScope = name;
        if (schematic) schematic.dataset.scope = name;
        choices.forEach(choice => {
          const active = choice.dataset.scope === name;
          choice.classList.toggle('is-active', active);
          choice.setAttribute('aria-pressed', String(active));
        });
        sharedModules.forEach(module => {
          module.classList.toggle('scope-trainable', scope.sharedTrainable);
          const state = module.querySelector('.scope-state');
          if (state) state.textContent = scope.sharedTrainable ? 'Trainable' : 'Frozen';
        });
        expertIcons.forEach((icon, index) => {
          icon.classList.toggle('scope-trainable', index < scope.symbolicTrainable);
        });
        experts?.setAttribute('aria-label', scope.expertLabel);
        if (expertStatus) expertStatus.textContent = scope.expertState;
        if (title) title.textContent = scope.title;
        if (description) description.textContent = scope.description;
        if (selectedScope) selectedScope.textContent = scope.title;
        if (ftGain) ftGain.textContent = `+${scope.ftGain.toFixed(2)}`;
        if (kdGain) kdGain.textContent = `+${scope.kdGain.toFixed(2)}`;
        // A shared 0–3.5 pp scale keeps the two visual indicators comparable.
        if (ftBar) ftBar.style.width = `${scope.ftGain / 3.5 * 100}%`;
        if (kdBar) kdBar.style.width = `${scope.kdGain / 3.5 * 100}%`;
      }

      choices.forEach(choice => choice.addEventListener('click', () => selectScope(choice.dataset.scope)));
      selectScope('router');
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
})();
