def get_mandatory_config(cfg, keys, file_context=""):
    """
    Mandatory retrieve parameter from config, raises detailed error if not found
    
    Args:
        cfg: Configuration object
        keys: Configuration key path, can be string or list
        file_context: File context information for error messages
        
    Returns:
        Configuration value
        
    Raises:
        KeyError: When configuration item does not exist
    """
    if isinstance(keys, str):
        keys = [keys]
    
    current = cfg
    path = []
    
    try:
        for key in keys:
            path.append(key)
            current = current[key]
        return current
    except KeyError as e:
        path_str = " -> ".join(path)
        raise KeyError(
            f"❌ Config item missing: '{path_str}'\n"
            f"📄 File: {file_context}\n"
            f"💡 Please add this config item to the config file"
        ) from e
    except TypeError as e:
        path_str = " -> ".join(path[:-1]) if len(path) > 1 else "root config"
        raise TypeError(
            f"❌ Config structure error: '{path_str}' is not a dict type\n"
            f"📄 File: {file_context}\n"
            f"💡 Please check the structure of config file"
        ) from e


def validate_config_section(cfg, section_key, required_fields, file_context=""):
    try:
        section = cfg[section_key]
    except KeyError:
        raise KeyError(
            f"❌ Config section missing: '{section_key}'\n"
            f"📄 File: {file_context}\n"
            f"💡 Please add this config section to the config file"
        )
    
    missing_fields = []
    for field in required_fields:
        if field not in section:
            missing_fields.append(field)
    
    if missing_fields:
        raise KeyError(
            f"❌ Config section '{section_key}' missing required fields: {missing_fields}\n"
            f"📄 File: {file_context}\n"
            f"💡 Please add these fields to the config file"
        )