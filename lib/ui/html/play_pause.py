play_pause_button = """
  <!-- Button to play/pause the audio -->
  <button class="gr-button gr-button-lg gr-button-secondary"
    onclick = "
      wavesurfer.playPause()
      this.innerText = wavesurfer.isPlaying() ? '⏸️' : '▶️'
    "
  >▶️</button>
  <!-- Textbox showing current time -->
  <input type="number" class="gr-box gr-input gr-text-input" id="audio-time" value="0">
  <!-- Download button -- it will be set to the right href later on -->
  <!--
  <a class="gr-button gr-button-lg gr-button-secondary" id="download-button">
    🔗
  </a>
  -->
  <!-- (Removed for now, as it only links to the first chunk, will fix later) -->
  <!-- Refresh button -- it virtually clicks the "internal-refresh-button" button (which is hidden) -->
  <button class="gr-button gr-button-lg gr-button-secondary" onclick="window.shadowRoot.getElementById('internal-refresh-button').click()" id="refresh-button">
    ↻
  </button>
"""