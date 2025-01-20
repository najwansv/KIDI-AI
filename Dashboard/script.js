document.getElementById('rtsp-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const rtspLink = document.getElementById('rtsp-link').value;
    const streamContainer = document.getElementById('stream-container');
    const videoPlayer = document.getElementById('rtsp-player');
    const mainContainer = document.getElementById('main-container');

    if (!rtspLink.startsWith('rtsp://')) {
        alert('Please enter a valid RTSP link.');
        return;
    }

    // Assuming a proxy or RTSPtoWebRTC server transforms the link
    const transformedLink = rtspLink.replace('rtsp://', 'https://your-proxy-server/');

    videoPlayer.src = transformedLink;

    // Show the stream container and hide the form
    streamContainer.classList.remove('hidden');
    mainContainer.querySelector('form').classList.add('hidden');
    mainContainer.querySelector('h1').classList.add('hidden');
});
