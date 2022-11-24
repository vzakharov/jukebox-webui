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

    Ji.getUnshownDuration = () => {
      // Take total duration from #total-audio-length's input
      let totalDuration = parseFloat(shadowSelector('#total-audio-length input').value)
      // console.log('Total duration: ', totalDuration)
      // Take preview duration from #preview-last-n-sec's input
      let previewDuration = parseFloat( shadowSelector('#preview-last-n-sec input')?.value ) || totalDuration
      // console.log('Preview duration: ', previewDuration)
      let unshownDuration = totalDuration - previewDuration
      // console.log('Unshown duration: ', unshownDuration)
      return unshownDuration
    }

    Ji.wavesurferToActualTime = wavesurferTime => {
      // We need this function because the preview audio can be shorter than the full audio, in which case we need to add the additional (non-shown) duration to the time
      return Math.round( ( wavesurferTime + Ji.getUnshownDuration() ) * 100 ) / 100
    }

    Ji.actualToWavesurferTime = actualTime => {
      // We need this function to know where to e.g. place certain markers in the preview audio
      return Math.round( ( actualTime - Ji.getUnshownDuration() ) * 100 ) / 100
    }

    let regionParams = {
      regions: [],
      dragSelection: false,
      maxRegions: 1,
      formatTimeCallback: Ji.wavesurferToActualTime,
      // Salmon color, but with opacity
      color: 'rgba(250, 128, 114, 0.2)',
    }

    window.wavesurfer = WaveSurfer.create({
      container: waveformDiv,
      waveColor: 'skyblue',
      progressColor: 'steelblue',
      plugins: [
        // markers (empty for now)
        WaveSurfer.markers.create(),
        // regions
        WaveSurfer.regions.create(regionParams),
      ]
    })

    Ji.addUpsamplingMarkers = ( times = Ji.upsamplingTimes ) => {
      Ji.upsamplingTimes = times
      wavesurfer.markers.clear()
      // delete all .upsampling-marker-tooltip elements
      document.querySelectorAll('.upsampling-marker-tooltip').forEach( el => el.remove() )
      if ( !times ) return
      times.slice().reverse().forEach( ( time, i ) => {
        time = Ji.actualToWavesurferTime(time)
        if ( time <= 0 ) return
        wavesurfer.markers.add({
          time,
          color: [ 'orange', 'lightgreen' ][i],
        }).el.querySelector('.marker-label').title = `Your audio has been ${[ 'midsampled', 'upsampled' ][i]} to this point.`
      } )
    }

    // Also add markers on every wavesurfer reload
    wavesurfer.on('ready', Ji.addUpsamplingMarkers)

    let cutAudioSpecsInput = shadowSelector('#cut-audio-specs textarea')
    let updatedAutomatically = false

    // On region update end, update the #cut-audio-specs input, isnerting [start]-[end] into the value
    wavesurfer.on('region-update-end', ({ start, end }) => {
      // We need to update value in such a way that any of the app's trigger events are fired
      // round to 2 decimal places
      cutAudioSpecsInput.value = [ start, end ].map( time => Math.round( Ji.wavesurferToActualTime(time) * 100 ) / 100 ).join('-')
      cutAudioSpecsInput.dispatchEvent(new Event('input'))
      updatedAutomatically = true
    })

    // Whenever that input changes, update the region:
    // - if the input is empty, remove the region
    // - if the input is formatted in any other way rather than [start]-[end], remove the region
    // - if the input is formatted correctly, update the region
    cutAudioSpecsInput.addEventListener('input', () => {

      if (updatedAutomatically) {
        updatedAutomatically = false
        return
      }

      // wavesurfer.regions.list is a hash not an array, so we need to get the only key starting with 'wavesurfer_'
      let { list } = wavesurfer.regions
      let region = list[ Object.keys(list).find( key => key.startsWith('wavesurfer_') ) ]
      let { value } = cutAudioSpecsInput

      // Check with regex if the input is formatted correctly
      if ( !value.match(/^\d+(\.\d+)?-\d+(\.\d+)?$/) ) {
        // If not, remove the region
        region?.remove()
        return
      }

      let [ start, end ] = cutAudioSpecsInput.value.split('-').map( time => Ji.actualToWavesurferTime(parseFloat(time)) )
      region ||= wavesurfer.addRegion(regionParams)
      region.update({ start, end })
    })

    // Remove the region on double click
    wavesurfer.on('region-dblclick', () => wavesurfer.clearRegions() )

    // Enable drags election when #cut-audio-specs input is focused, disable when blurred
    cutAudioSpecsInput.addEventListener('focus', () => wavesurfer.regions.enableDragSelection() )
    cutAudioSpecsInput.addEventListener('blur', () => wavesurfer.regions.disableDragSelection() )
    
    Ji.trackTime = time => (
      shadowSelector('#audio-time').value = Ji.wavesurferToActualTime(time),
      Ji.currentTime = time
    )

    // Also update the time when the audio is playing
    wavesurfer.on('audioprocess', Ji.trackTime)
    // Add a seek event listener to the wavesurfer object, modifying the #audio-time input
    wavesurfer.on('seek', progress => Ji.trackTime(progress * wavesurfer.getDuration()))

    // When wavesurfer starts/stops playing, update Ji.playing
    wavesurfer.on('play', () => Ji.playing = true)
    wavesurfer.on('pause', () => Ji.playing = false)
    
    Ji.grayOutWavesurfer = ( on = true ) => {
      // Gray out the wavesurfer (e.g. when the audio is being refreshed)
      let { container } = wavesurfer
      if ( on ) {
        container.style.filter = 'grayscale(1)'
      } else {
        container.style.filter = ''
      }
    }

    wavesurfer.on('ready', () => {

      // Stop the clock blinking
      clearInterval(Ji.clockInterval)

      // Seek to the remembered time, unless it's higher than the new audio length
      let duration = wavesurfer.getDuration()
      Ji.currentTime < duration && wavesurfer.seekTo(Ji.currentTime / duration)

      // Start playing if Ji.playing is true
      Ji.playing && wavesurfer.play()
      
      // Replace the clock with a refresh glyph
      if ( Ji.refreshButton ) {
        Ji.refreshButton.innerText = '↻'
      }

      // Remove the gray filter, unless it was preloaded, in which case we'll do this once we make sure the loaded audio is the same as the preloaded one
      if ( !Ji.preloadedAudio ) {
        Ji.grayOutWavesurfer(false)
      }

    })

    Ji.blobCache = []

    Ji.maxCacheSize = 100

    Ji.addBlobToCache = ( key, blob ) => {
      let { blobCache, maxCacheSize } = Ji
      // If there's >= Ji.maxCacheSize elements in the cache, remove the oldest one
      if ( blobCache.length >= maxCacheSize ) {
        blobCache.pop()
      }
      blobCache.unshift({ key, blob })
      console.log(`Added ${key} to cache, total cache size: ${blobCache.length} (${blobCache.reduce( (acc, { blob }) => acc + blob.size, 0 )/1000000} MB)`)
    }

    Ji.fetchBlob = async element => {
      console.log(`Fetching blob for ${element.href}`)
      let response = await fetch(element.href)
      let blob = await response.blob()
      console.log(`Loaded blob:`, blob)
      return blob
    }

    Ji.blobSHA = blob => {
      return new Promise( resolve => {
        let reader = new FileReader()
        reader.onload = () => {
          let arrayBuffer = reader.result
          // use javascript's built in crypto library to create a sha1 hash of the arrayBuffer
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

    Ji.getFilenameFromHref = href => href
      // Remove path & extension
      .replace(/^.*\//, '').replace(/\.[^/.]+$/, '')
      // and replace all %20's with spaces
      .replace(/%20/g, ' ')
      // and remove everything after and including the last space (this is where Gradio puts the hash)
      .replace(/ .*$/, '')
      // and \s\d+-\d+ in the end, indicating a chunk (if any)
      .replace(/\s\d+-\d+$/, '')
      // trim
      .trim()

    Ji.reloadAudio = async () => {

      Ji.audioElements = Ji.currentChunksContainer.querySelectorAll('a')

      let audioHref = Ji.audioElements[0].href

      if ( audioHref == Ji.lastAudioHref ) {
        console.log('The audio is already loaded, not reloading')
        return
      }
      
      let filename = Ji.getFilenameFromHref(audioHref)

      if ( filename == Ji.lastLoadedBlobKey ) {
        console.log(`The blob for ${filename} is already loaded, not reloading`)
        Ji.grayOutWavesurfer(false)
        return
      }

      console.log(`The audio is not loaded, reloading from ${audioHref}`)

      // Replace the #reload-button inner text with an clock, blinking with different times at 0.5s intervals
      Ji.refreshButton = shadowRoot.querySelector('#refresh-button')
      if ( Ji.refreshButton ) {
        let emojis = [ '🕛', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚' ]
        let flip = () => emojis.push( Ji.refreshButton.innerText = emojis.shift() )      
        clearInterval(Ji.clockInterval)
        Ji.clockInterval = setInterval( flip, 500 )
        flip()
      }

      console.log(`Checking blob cache for ${filename}`)
      let cachedBlob = Ji.blobCache.find( ({ key }) => key == filename )

      if ( cachedBlob && cachedBlob instanceof Promise ) {
        console.log(`Blob for ${filename} is a promise, waiting for it to resolve`)
        cachedBlob = await cachedBlob
      }        

      // If cached blob is a promise, wait for it to resolve
      // If not cached, combine a blob from all the chunks in audioElements
      let blob = cachedBlob?.blob || new Blob( Ji.mainBlobPromise = await Promise.all( Array.from(Ji.audioElements).map( Ji.fetchBlob ) ), { type: 'audio/mpeg' } )
      
      wavesurfer.loadBlob(blob)

      !cachedBlob && Ji.addBlobToCache( filename, blob )

      window.shadowRoot.querySelector('#download-button').href = audioHref

      Ji.lastAudioHref = audioHref
      
    }

    // Put an observer on #current-chunks (also in the shadow DOM) to reload the audio from its inner <a> element
    Ji.currentChunksContainer = window.shadowRoot.querySelector('#current-chunks')

    let currentChunksObserver = new MutationObserver( () => {
      
      // Check if there is an inner <a> element
      Ji.audioElements = Ji.currentChunksContainer.querySelectorAll('a')

      if ( Ji.audioElements.length ) {

        // Once we've foun audio elements, we can remove the parent observer and create one for the first audio element instead
        // This element contains the actual audio file (the others contain chunks of it for faster loading), so whenever the first one changes, we need to reload wavesurfer
        
        console.log('Found audio elements, removing parent observer')

        currentChunksObserver.disconnect()
        Ji.lastAudioHref = null

        // Reload the audio at once
        Ji.reloadAudio()
        

        // And also reload it whenever the audio href changes
        new MutationObserver(Ji.reloadAudio).observe(Ji.audioElements[0], { attributes: true, attributeFilter: ['href'] })

      }
        
    })

    currentChunksObserver.observe(Ji.currentChunksContainer, { childList: true, subtree: true })

    // Put an observer on #sibling-chunks to cache the audio from its inner <a> elements
    Ji.siblingChunksContainer = window.shadowRoot.querySelector('#sibling-chunks')

    let siblingChunksObserver = new MutationObserver( () => {

      // Check if there is an inner <a> element
      let audioElements = Ji.siblingChunksContainer.querySelectorAll('a')

      if ( audioElements.length ) {
        
        // Once we've found audio elements, we can remove the parent observer and create one for the first audio element instead
        console.log('Found sibling audio elements, removing parent observer')

        siblingChunksObserver.disconnect()
        
        let cacheSiblingChunks = async () => {

          console.log('Sibling chunk container updated, let’s cache some blobs...')

          await Ji.mainBlobPromise
          // (We need to wait for the main blob to be loaded, so that the others don't slow down the loading of the main one)

          console.log('Main blob loaded, caching...')

          let audioElements = Ji.siblingChunksContainer.querySelectorAll('a')

          let blobPromisesByFilename = {}
          // Make sure no more than 10 requests are sent per second
          let rateLimit = 100
          let totalDelay = 0

          for ( let audioElement of audioElements ) {

            let filename = Ji.getFilenameFromHref(audioElement.href)

            // If cached, skip
            if ( Ji.blobCache.find( ({ key }) => key == filename ) )
              continue

            let chunkPromise = new Promise( async resolve => {
              // Wait for totalDelay (which is increased by rateLimit each time)
              // totalDelay += rateLimit
              let delay = totalDelay += rateLimit
              await new Promise( resolve => setTimeout(resolve, delay) )
              console.log(`Fetching ${filename} after ${delay}ms`)
              let blob = await Ji.fetchBlob(audioElement)
              console.log(`Fetched ${filename}:`, blob)
              resolve(blob)
            } )

            ;( blobPromisesByFilename[filename] ||= [] ).push( chunkPromise )

          }

          // Combine all the blobs for each filename (await Promise.all for faster loading)
          await Promise.all( Object.entries(blobPromisesByFilename).map( async ([filename, blobPromises]) => {
            let blob = new Blob(await Promise.all(blobPromises), { type: 'audio/mpeg' })
            Ji.addBlobToCache( filename, blob )
            console.log(`Cached blob for ${filename}`)
          } ) )

        }

        // Cache the audio at once
        cacheSiblingChunks()

        // And also cache it whenever the audio href changes
        new MutationObserver(cacheSiblingChunks).observe(audioElements[0], { attributes: true, attributeFilter: ['href'] })

      }

    })

    siblingChunksObserver.observe(Ji.siblingChunksContainer, { childList: true, subtree: true })

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