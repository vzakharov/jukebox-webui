# Jukebox Web UI: Getting started

([permalink](https://github.com/vzakharov/jukebox-webui/blob/main/docs/getting-started.md))

## Steps

1. Create a project

2. Set up the generation parameters

3. Click Generate to generate several samples of the first few seconds of the song

4. Continue to generate more samples from the ones you like, until you have a finished song

## Managing expectations

### You don’t get an entire song in one click

Neither this app nor Jukebox is meant to generate entire songs at once. You go through a process of generating samples, listening to them, and selecting the ones you like.

### You don’t get to generate individual tracks

Jukebox is a model that generates the “master track” of a song. You can’t tell it to generate a bassline, a drum track, or a melody to later combine into a song. You can only generate the song as a whole.

### It takes time

Even with the lowest quality (the only one that is currently supported), it takes 30-60 seconds to generate ONE second of music.

### The quality is, well...

Before upsampling (which is not currently supported), the quality is just enough to be able to discern. Even after upsampling (which takes around 10 hours per minute of music), the quality will not be as good as a professional recording. Something like [this](https://www.youtube.com/watch?v=xeJesnxvKB0&list=PLhW3E8TjBWfjs3DK57_FmPv9Ag62WGxNJ&index=3) is the best you can expect.

### I have no idea what I’m doing

Last but not least, this app is something I use myself, so I never intended to make it super user-friendly or foolproof. Neither do I have enough resources to provide support. Luckily for you, the whole thing is open source, so you can always look at the code and figure out what’s going on -- or at least try and find someone who is more responsive than me.

All that being said, I still think that this app is a lot of fun, and opens up a lot of exciting new possibilities for music creation. I hope you’ll enjoy it as much as I do.

Have fun, and happy music making!

~ [Vova](https://twitter.com/vovahimself)

## FAQ

**Q: What GPU do I need?**

A: A Tesla T4 GPU will do just fine, so you can choose “Standard” in your Colab settings. Note that using a high-end GPU such as an A100 will give just around 10% better quality, while costing ~7 times more.

**Q: Can I run this on my own machine?**

A: You probably can if you manage to install Jukebox and have a powerful enough GPU. I have never tried, so this will be a bit of an adventure for you. Make sure to comment out all lines that start with `!` in `jukebox-webui.py` before running it, as those are notebook-specific commands.

## Roadmap

- [x] v0.1: Initial release
- [ ] v0.2: Add support for using your own audio as a starting point
- [ ] v0.3: Add support for upsampling