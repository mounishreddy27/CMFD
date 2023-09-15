$(function() {
    // Set the cookie when the page is loaded
    document.cookie = "session_expire_at_browser_close=true";

    // Remove the cookie when the page is unloaded (browser close)
    $(window).on("unload", function() {
        document.cookie = "session_expire_at_browser_close=; expires=Thu, 01 Jan 1970 00:00:01 GMT;";
        clearMediaFiles();
    });

    function clearMediaFiles() {
        $.ajax({
            url: '/delete_media_files/',
            type: 'POST',
            data: {csrfmiddlewaretoken: '{{ csrf_token }}'},
            success: function(response) {
                console.log(response);
            },
            error: function(response) {
                console.log(response);
            }
        });
    }
});