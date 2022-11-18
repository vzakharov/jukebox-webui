async () => {

  // The function is called by the `load` event handler in the python backend (Gradio)
      
  try {

    // Create and inject wavesurfer scripts
    let require = url => {
      console.log(`Injecting ${url}`)
      let script = document.createElement('script')
      script.src = url
      document.head.appendChild(script)
      return new Promise( resolve => script.onload = () => {
        console.log(`Injected ${url}`)
        resolve()
      } )
    }

    await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/wavesurfer.min.js')
    await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/plugin/wavesurfer.timeline.min.js')

    window.shadowRoot = document.querySelector('gradio-app').shadowRoot

    // The wavesurfer element is hidden inside a shadow DOM hosted by <gradio-app>, so we need to get it from there
    let shadowSelector = selector => window.shadowRoot.querySelector(selector)

    let waveformDiv = shadowSelector('#audio-waveform')
    console.log(`Found waveform div:`, waveformDiv)

    let timelineDiv = shadowSelector('#audio-timeline')
    console.log(`Found timeline div:`, timelineDiv)
    
    let getAudioTime = time => {
      let previewDuration = wavesurfer.getDuration()
      // console.log('Preview duration: ', previewDuration)
      // Take total duration from #total-audio-length's input, unless #trim-to-n-sec is set, in which case use that
      let trimToNSec = parseFloat(shadowSelector('#trim-to-n-sec input')?.value || 0)
      // console.log('Trim to n sec: ', trimToNSec)
      let totalDuration = trimToNSec || parseFloat(shadowSelector('#total-audio-length input').value)
      // console.log('Total duration: ', totalDuration)
      let additionalDuration = totalDuration - previewDuration
      // console.log('Additional duration: ', additionalDuration)
      let result = Math.round( ( time + additionalDuration ) * 100 ) / 100          
      // console.log('Result: ', result)
      return result
    }

    // Create a (global) wavesurfer object with and attach it to the div
    window.wavesurferTimeline = WaveSurfer.timeline.create({
      container: timelineDiv,
      // Light colors, as the background is dark
      primaryColor: '#eee',
      secondaryColor: '#ccc',
      primaryFontColor: '#eee',
      secondaryFontColor: '#ccc',
      formatTimeCallback: time => Math.round(getAudioTime(time))
    })

    window.wavesurfer = WaveSurfer.create({
      container: waveformDiv,
      waveColor: 'skyblue',
      progressColor: 'steelblue',
      plugins: [
        window.wavesurferTimeline
      ]
    })
    
    // Add a seek event listener to the wavesurfer object, modifying the #audio-time input
    wavesurfer.on('seek', progress => {
      shadowSelector('#audio-time').value = getAudioTime(progress * wavesurfer.getDuration())
    })

    // Also update the time when the audio is playing
    wavesurfer.on('audioprocess', time => {
      shadowSelector('#audio-time').value = getAudioTime(time)
    })

    // Put an observer on #audio-file (also in the shadow DOM) to reload the audio from its inner <a> element
    let parentElement = window.shadowRoot.querySelector('#audio-file')
    let parentObserver = new MutationObserver( () => {
      
      // Check if there is an inner <a> element
      let audioElements = parentElement.querySelectorAll('a')

      if ( audioElements.length ) {

        // Once we've foun audio elements, we can remove the parent observer and create one for the first audio element instead
        // This element contains the actual audio file (the others contain chunks of it for faster loading), so whenever the first one changes, we need to reload wavesurfer
        
        console.log('Found audio elements, removing parent observer')

        parentObserver.disconnect()
        lastAudioHref = null

        let reloadAudio = async () => {

          console.log('Audio element updated, checking if href changed...')
          audioElements = parentElement.querySelectorAll('a')

          let audioHref = audioElements[0].href

          if ( audioHref == lastAudioHref ) {
            console.log('Audio href has not changed, skipping.')
            return
          }

          console.log('Audio href has changed, reloading audio...')
          let loadBlob = async ( href, isPartial ) => {
            let response = await fetch(href)
            let blob = await response.blob()
            console.log(`Loaded ${isPartial ? 'partial ' : ''}audio blob:`, blob)
            return blob
          }

          let blob

          // If there are several audio elements, load the ones starting with the second one as blobs and add them to the wavesurfer object
          if ( audioElements.length > 1 ) {
            
            let audioBlobPromises = []

            for ( let i = 1; i < audioElements.length; i++ ) {
              let audioElement = audioElements[i]
              audioBlobPromises.push( loadBlob(audioElement.href, true) )
              audioBlobPromises.push(audioBlobPromise)
            }
            Promise.all(audioBlobPromises).then( audioBlobs => {                  
              // Combine the audio blobs into a single blob
              blob = new Blob(audioBlobs, { type: 'audio/mpeg' })
              console.log('Combined audio blob:', blob)
            } )

          } else {
            // If there is only one audio element, load it directly
            blob = await loadBlob(audioHref)
          }

          // Load the blob into wavesurfer
          wavesurfer.loadBlob(blob)

          window.shadowRoot.querySelector('#download-button').href = audioElements[0].href

          previousAudioHrefs = audioHrefs

        }

        // Reload the audio at once
        reloadAudio()

        // And also reload it whenever the audio href changes
        new MutationObserver(reloadAudio).observe(audioElements[0], { attributes: true, attributeFilter: ['href'] })

      }
        
    })

    parentObserver.observe(parentElement, { childList: true, subtree: true })

    window.Ju = {}

    Ju.clickTabWithText = function (buttonText) {
      for ( let button of document.querySelector('gradio-app').shadowRoot.querySelectorAll('div.tabs > div > button') ) {
        if ( button.innerText == buttonText ) {
          button.click()
          break
        }
      }
    }

    // href, query_string, error_message
    return [ window.location.href, window.location.search.slice(1), null ]

  } catch (e) {

    console.error(e)

    // If anything went wrong, return the error message
    return [ window.location.href, window.location.search.slice(1), e.message ]

  }
}