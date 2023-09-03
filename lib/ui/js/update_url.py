update_url_js = '''
    async ( ...args ) => {
      try {
        let sample_id = args[1]
        sample_id && window.history.pushState( {}, '', `?${args[1]}` )
        // Gray out the wavesurfer
        Ji.grayOutWavesurfer()
        // Now we'll try to reload the audio from cache. To do that, we'll find the first cached blob (Ji.blobCache) whose key starts with the sample_id either followed by space or end of string.
        // (Although different version of the same sample might have been cached, the first one will be the one that was added last, so it's the most recent one)
        let cached_blob = Ji.blobCache.find( ({ key }) => key.match( new RegExp(`^${sample_id}( |$)`) ) )
        if ( cached_blob ) {
          console.log( 'Found cached blob', cached_blob )
          let { key, blob } = cached_blob
          wavesurfer.loadBlob( blob )
          Ji.lastLoadedBlobKey = key
          Ji.preloadedAudio = true
          // Gray out slightly less
          Ji.grayOutWavesurfer( true, 0.75 )
        }
      } catch (e) {
        console.error(e)
      } finally {
        return args
      }
    }
  '''