function handleFileUpload(e) {
    e.preventDefault();

    const files = document.getElementById('upload-file').files;

    if (files.length === 0) {
        console.error('Runtime error occured: no file is selected');
    }

    const formData = new FormData();
    formData.append('song', files[0]);

    const requestBody = {
        method: 'POST',
        body: formData,
    }

    fetch('song', requestBody).then((response) => {
        response.json().then((json) => {
            console.log(json)
        });
    });
}

// Add event listener to handle when a file is selected
document.getElementById('upload-file').addEventListener('change', handleFileUpload);

// Add event listener to upload-button and make it trigger the file dialog
document.getElementById('upload-button').addEventListener('click', () => {
    document.getElementById('upload-file').click();
});