async () => {

  // The function is called by the `load` event handler in the python backend (Gradio)

  window.Ji = {}

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

    // import wavesurfer
    await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/wavesurfer.min.js')
    // import wavesurfer markers plugin
    await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/plugin/wavesurfer.markers.min.js')
    // regions
    await require('https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/6.3.0/plugin/wavesurfer.regions.min.js')

    window.shadowRoot = document.querySelector('gradio-app').shadowRoot

    // The wavesurfer element is hidden inside a shadow DOM hosted by <gradio-app>, so we need to get it from there
    window.shadowSelector = selector => window.shadowRoot.querySelector(selector)

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

    window.wavesurfer = WaveSurfer.create({
      container: waveformDiv,
      waveColor: 'skyblue',
      progressColor: 'steelblue',
      plugins: [
        // markers (empty for now)
        WaveSurfer.markers.create(),
        // regions
        WaveSurfer.regions.create({
          regions: [],
          dragSelection: true,
          maxRegions: 1,
          formatTimeCallback: getAudioTime,
        })
      ]
    })

    Ji.addUpsamplingMarkers = times => {
      wavesurfer.markers.clear()
      // delete all .upsampling-marker-tooltip elements
      document.querySelectorAll('.upsampling-marker-tooltip').forEach( el => el.remove() )
      times.reverse().forEach( ( time, i ) => {
        if (!time) return
        wavesurfer.markers.add({
          time,
          color: [ 'orange', 'lightgreen' ][i],
          label: [ 'M', 'U' ][i],
          // tooltip: `Your audio has been ${[ 'midsampled', 'upsampled' ][i]} to this point (${getAudioTime(time)} s)`,
          // For some reason tooltips don't work at all, we'll need to write our own
        }).el.querySelector('.marker-label').title = `Your audio has been ${[ 'midsampled', 'upsampled' ][i]} to this point (${getAudioTime(time)} s)`
      } )
    }

    let cutAudioSpecsInput = shadowSelector('#cut-audio-specs textarea')

    // On region update end, update the #cut-audio-specs input, isnerting [start]-[end] into the value
    wavesurfer.on('region-update-end', ({ start, end }) => {
      // We need to update value in such a way that any of the app's trigger events are fired
      // round to 2 decimal places
      cutAudioSpecsInput.value = [ start, end ].map( time => Math.round( time * 100 ) / 100 ).join('-')
      cutAudioSpecsInput.dispatchEvent(new Event('input'))
    })

    // Whenever that input changes, update the region:
    // - if the input is empty, remove the region
    // - if the input is formatted in any other way rather than [start]-[end], remove the region
    // - if the input is formatted correctly, update the region
    cutAudioSpecsInput.addEventListener('input', () => {
      let [ start, end ] = cutAudioSpecsInput.value.split('-').map( time => parseFloat(time) )
      let existingRegion = wavesurfer.regions.list[0]
      if (isNaN(start) || isNaN(end)) {
        existingRegion?.remove()
      } else {
        existingRegion?.update({ start, end }) || wavesurfer.addRegion({ start, end })
      }
    })

    
    Ji.trackTime = time => (
      shadowSelector('#audio-time').value = getAudioTime(time),
      Ji.currentTime = time
    )

    // Also update the time when the audio is playing
    wavesurfer.on('audioprocess', Ji.trackTime)
    // Add a seek event listener to the wavesurfer object, modifying the #audio-time input
    wavesurfer.on('seek', progress => Ji.trackTime(progress * wavesurfer.getDuration()))

    // When wavesurfer starts/stops playing, update Ji.playing
    wavesurfer.on('play', () => Ji.playing = true)
    wavesurfer.on('pause', () => Ji.playing = false)

    // Put an observer on #audio-file (also in the shadow DOM) to reload the audio from its inner <a> element
    Ji.parentAudioElement = window.shadowRoot.querySelector('#audio-file')

    Ji.blobCache = []

    Ji.maxCacheSize = 1000

    Ji.addBlobToCache = ( key, blob ) => {
      let { blobCache, maxCacheSize } = Ji
      // If there's >= Ji.maxCacheSize elements in the cache, remove the oldest one
      if ( blobCache.length >= maxCacheSize ) {
        blobCache.shift()
      }
      blobCache.push({ key, blob })
      console.log(`Added ${key} to cache, total cache size: ${blobCache.length} (${blobCache.reduce( (acc, { blob }) => acc + blob.size, 0 )/1000000} MB)`)
    }

    Ji.blobSHA = blob => {
      return new Promise( resolve => {
        let reader = new FileReader()
        reader.onload = () => {
          let arrayBuffer = reader.result
          // use javascrit's built in crypto library to create a sha1 hash of the arrayBuffer
          crypto.subtle.digest('SHA-1', arrayBuffer).then( hashBuffer => {
            // convert the hashBuffer to a hex string
            let hashArray = Array.from(new Uint8Array(hashBuffer))
            let hashHex = hashArray.map( b => b.toString(16).padStart(2, '0') ).join('')
            resolve(hashHex)
          })
        }
        reader.readAsArrayBuffer(blob)
      })
    }        


    let parentObserver = new MutationObserver( () => {
      
      // Check if there is an inner <a> element
      Ji.audioElements = Ji.parentAudioElement.querySelectorAll('a')

      if ( Ji.audioElements.length ) {

        // Once we've foun audio elements, we can remove the parent observer and create one for the first audio element instead
        // This element contains the actual audio file (the others contain chunks of it for faster loading), so whenever the first one changes, we need to reload wavesurfer
        
        console.log('Found audio elements, removing parent observer')

        parentObserver.disconnect()
        lastAudioHref = null

        Ji.reloadAudio = async () => {

          console.log('Audio element updated, checking if href changed...')
          Ji.audioElements = Ji.parentAudioElement.querySelectorAll('a')

          let audioHref = Ji.audioElements[0].href

          if ( audioHref == lastAudioHref ) {
            console.log('Audio href has not changed, skipping.')
            return
          }

          console.log(`Audio href changed to ${audioHref}, reloading wavesurfer...`)

          // Replace the #reload-button inner text with an hourglass
          let refreshButton = shadowRoot.querySelector('#refresh-button')
          if ( refreshButton ) {
            refreshButton.innerText = '⏳'
          }

          let loadBlob = async element => {
            console.log(`Fetching blob for ${filename}`)
            let response = await fetch(element.href)
            let blob = await response.blob()
            console.log(`Loaded blob:`, blob)
            return blob
          }

          // Remove path & extension
          let filename = audioHref.replace(/^.*\//, '').replace(/\.[^/.]+$/, '')
          

          console.log(`Checking blob cache for ${filename}`)
          let cachedBlob = Ji.blobCache.find( ({ key }) => key == filename )

          let blob = cachedBlob?.blob || (
            Ji.audioElements.length > 1 ?
              new Blob(await Promise.all( Array.from(Ji.audioElements).slice(1).map( loadBlob ) ), { type: 'audio/mpeg' }) :
              await loadBlob(Ji.audioElements[0])
          )
          
          // compare the preloaded blob's SHA to the one in the cache
          let blobSHA = await Ji.blobSHA(blob)
          if ( blobSHA != Ji.preloadedBlobSHA ) {
            console.log(`Blob SHA changed to ${blobSHA}, reloading wavesurfer...`)
            wavesurfer.loadBlob(blob)

            Ji.preloadedBlobKey && Ji.addBlobToCache( Ji.preloadedBlobKey, blob )
            
            wavesurfer.on('ready', () => {

              // Seek to the remembered time, unless it's higher than the new audio length
              let duration = wavesurfer.getDuration()
              Ji.currentTime < duration && wavesurfer.seekTo(Ji.currentTime / duration)

              // Start playing if Ji.playing is true
              Ji.playing && wavesurfer.play()
              
              // Replace the hourglass with a refresh glyph
              if ( refreshButton ) {
                refreshButton.innerText = '↻'
              }

            })

          } else {
            console.log('Blob SHA has not changed, skipping.')
          }

          !cachedBlob && Ji.addBlobToCache( filename, blob )

          window.shadowRoot.querySelector('#download-button').href = audioHref

          lastAudioHref = audioHref
          
        }

        // Reload the audio at once
        Ji.reloadAudio()

        // And also reload it whenever the audio href changes
        new MutationObserver(Ji.reloadAudio).observe(Ji.audioElements[0], { attributes: true, attributeFilter: ['href'] })

      }
        
    })

    parentObserver.observe(Ji.parentAudioElement, { childList: true, subtree: true })

    Ji.clickTabWithText = function (buttonText) {
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