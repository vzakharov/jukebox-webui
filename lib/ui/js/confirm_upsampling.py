confirm_upsampling_js = """
    // Confirm before starting/stopping the upsample process
    running => {
      confirmText = 
        running ?
          'Are you sure you want to stop the upsample process? ⚠️ THIS WILL KILL THE RUNTIME AND YOU WILL HAVE TO RESTART IT IN COLAB ⚠️ (But your current upsampling progress will be saved)' :
          'Are you sure you want to start the upsample process? THIS WILL TAKE HOURS, AND YOU WON’T BE ABLE TO CONTINUE COMPOSING!'
      if ( !confirm(confirmText) ) {
        throw new Error(`${running ? 'Stopping' : 'Starting'} upsample process canceled by user`)
      } else {
        // If running, show a message saying to restart the runtime in Colab
        if ( running ) {
          alert('Upsample process stopped. Please re-run the cell in Colab to restart the UI')
        }
        return [ running ]
      }
    }
  """